"""Diffusion Forcing training module for temporal single-cell generation."""

import math
from typing import Any, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.optim.lr_scheduler import LambdaLR

from cellpace.utils.logging import setup_logger

from .diffusion import Diffusion

logger = setup_logger("dif_core")


class DiffusionForcing(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.arch_cfg = cfg.architecture
        self.diffusion_cfg = cfg.diffusion
        self.generation_cfg = cfg.generation
        self.training_cfg = cfg.training

        self.x_shape = (self.arch_cfg.input_dim,)
        self.x_stacked_shape = list(self.x_shape)
        self.x_stacked_shape[0] *= self.arch_cfg.frame_stack

        self._configure_loss_weighting_strategy()

        self.generation_outputs = []
        self._build_model()

    def _configure_loss_weighting_strategy(self) -> None:
        """
        Configure loss weighting based on high-level strategy selector.

        Only Min-SNR weighting is supported.
        """
        # Get strategy from config
        strategy = self.training_cfg.loss.strategy

        # Get loss parameters
        snr_clip = self.training_cfg.loss.snr_clip

        if strategy == "snr":
            self.snr_clip = snr_clip
            logger.info(f"Loss weighting: Min-SNR (clip={snr_clip})")
        else:
            raise NotImplementedError(
                f"Loss weighting strategy '{strategy}' is not implemented. "
                f"Only 'snr' (Min-SNR) is supported."
            )

        self.loss_weighting_strategy = strategy

    def register_data_mean_std(
        self,
        mean: str | float | Sequence,
        std: str | float | Sequence,
        namespace: str = "data",
    ) -> None:
        """
        Register mean and std of data as tensor buffer.

        Args:
            mean: the mean of data.
            std: the std of data.
            namespace: the namespace of the registered buffer.
        """
        for k, v in [("mean", mean), ("std", std)]:
            if isinstance(v, str):
                if v.endswith(".npy"):
                    v = torch.from_numpy(np.load(v))
                elif v.endswith(".pt"):
                    v = torch.load(v, weights_only=False)
                else:
                    raise ValueError(f"Unsupported file type {v.split('.')[-1]}.")
            else:
                v = torch.tensor(v)
            self.register_buffer(f"{namespace}_{k}", v.float().to(self.device))

    def _build_model(self) -> None:
        self.diffusion_model = Diffusion(
            x_shape=torch.Size(self.x_stacked_shape),
            external_cond_dim=self.arch_cfg.external_cond_dim,
            is_causal=self.arch_cfg.causal,
            diffusion_cfg=self.cfg.diffusion,
            transformer_cfg=self.cfg.architecture.transformer,
            gap_config=self.cfg.architecture.gap_modeling,
            seq_len=self.cfg.architecture.sequence_length,
            snr_clip=self.snr_clip,
        )

    def configure_optimizers(self) -> dict[str, Any]:
        params = tuple(self.diffusion_model.parameters())
        opt_cfg = self.training_cfg.optimizer
        optimizer = torch.optim.AdamW(
            params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=opt_cfg.betas,
        )

        # Use Lightning's CosineAnnealingWarmRestarts or custom LambdaLR
        warmup_steps = opt_cfg.warmup_steps
        base_lr = opt_cfg.lr
        lr_min = opt_cfg.lr_min
        max_steps = self.training_cfg.max_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Linear warmup
                return float(step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing after warmup
                progress = float(step - warmup_steps) / float(
                    max(1, max_steps - warmup_steps)
                )
                return (lr_min / base_lr) + (1 - lr_min / base_lr) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update every step, not epoch
                "frequency": 1,
            },
        }

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        sequences = batch["sequence"]
        stage_positions = batch["stage_positions"]
        time_gaps = batch.get("time_gaps", None)

        xs = sequences.transpose(0, 1)
        stage_positions = stage_positions.transpose(0, 1)
        time_gaps = time_gaps.transpose(0, 1)

        temporal_features = torch.stack(
            [stage_positions, time_gaps], dim=-1
        )

        masks = torch.ones_like(stage_positions)

        xs_pred, loss = self.diffusion_model(
            xs,
            stage_positions=temporal_features,
            external_cond=None,
            noise_levels=self._generate_noise_levels(xs),
        )
        loss = self.reweight_loss(loss, masks)

        # Lightning best practice: always log, let logger handle frequency
        self.log(
            "train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )

        # Log learning rate
        if self.trainer.optimizers:
            self.log(
                "train/lr",
                self.trainer.optimizers[0].param_groups[0]["lr"],
                on_step=True,
                on_epoch=False,
                logger=True,
            )

        # No unnormalization needed - just unstack
        xs = rearrange(
            xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.arch_cfg.frame_stack
        )
        xs_pred = rearrange(
            xs_pred, "t b (fs c) ... -> (t fs) b c ...", fs=self.arch_cfg.frame_stack
        )

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    def _generate_noise_levels(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Generate noise levels for training.
        """
        num_frames, batch_size, *_ = xs.shape
        match self.generation_cfg.noise_level:
            case "random_all":  # entirely random noise levels
                noise_levels = torch.randint(
                    0,
                    self.diffusion_cfg.timesteps,
                    (num_frames, batch_size),
                    device=xs.device,
                )
            case _:
                # Default to random_all
                noise_levels = torch.randint(
                    0,
                    self.diffusion_cfg.timesteps,
                    (num_frames, batch_size),
                    device=xs.device,
                )

        return noise_levels

    def reweight_loss(
        self, loss: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        loss = rearrange(
            loss, "t b (fs c) ... -> t b fs c ...", fs=self.arch_cfg.frame_stack
        )
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape) - 1
            weight = rearrange(
                weight,
                "(t fs) b ... -> t b fs ..." + " 1" * expand_dim,
                fs=self.arch_cfg.frame_stack,
            )
            loss = loss * weight

        return loss.mean()
