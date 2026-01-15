"""Train and generate commands for CellPace CLI."""

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import scvi
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from cellpace import (
    DataManager,
    DiffusionForcing,
    DiFInference,
    GenerationConfig,
    GenerationData,
    ModelConfig,
    log_print,
    train_multivi,
    train_scvi,
)
from cellpace.model.vae.base import VAEUtils
from cellpace.utils.logging import close_logging, setup_logging


def load_vae_model(
    vae_type: str,
    vae_dir: Path,
    data_manager: Any,
    data_split: str = "train"
) -> scvi.model.SCVI | scvi.model.MULTIVI:
    """
    Unified VAE model loading.

    Args:
        vae_type: "scvi" or "multivi"
        vae_dir: Path to VAE model directory
        data_manager: DataManager instance
        data_split: "train" or "test"

    Returns:
        vae_model: Loaded scVI or MultiVI model
    """
    vae_dir = Path(vae_dir)

    if vae_type == "multivi":
        adata = getattr(data_manager, f"{data_split}_mdata")
        model = scvi.model.MULTIVI.load(str(vae_dir), adata=adata)
    else:
        adata = getattr(data_manager, f"{data_split}_adata")
        model = scvi.model.SCVI.load(str(vae_dir), adata=adata)

    return model


def train(config: DictConfig | ListConfig, model_type: str) -> None:
    """Train scVI (Stage 1) or DiF (Stage 2) model."""
    data_cfg = config.data
    exp_cfg = config.experiment
    vae_type = exp_cfg.vae_to_use
    data_manager = DataManager(data_cfg, seed=exp_cfg.seed, vae_type=vae_type)

    if model_type == "scvi":
        log_print(f"\nTraining scVI model to {exp_cfg.vae_dir}")
        model_path = train_scvi(config, data_manager)
        log_print(f"\nscVI training complete! Model saved to: {model_path}")
        return
    elif model_type == "multivi":
        log_print(f"\nTraining MultiVI model to {exp_cfg.vae_dir}")
        model_path = train_multivi(config, data_manager)
        log_print(f"\nMultiVI training complete! Model saved to: {model_path}")
        return
    elif model_type == "dif":
        log_print(f"\nTraining DiF model to {exp_cfg.dif_dir}")
        dif_cfg = config.dif
        logging_cfg = config.logging
        vae_model_path = Path(exp_cfg.vae_dir)
        log_print(f"Loading {vae_type.upper()} model from {vae_model_path}")

        vae_model = load_vae_model(vae_type, vae_model_path, data_manager, "train")
        vae_hash = VAEUtils.get_model_hash(vae_model_path)
        log_print(f"{vae_type.upper()} model hash: {vae_hash}")

        # MultiVI uses joint latents combining RNA+ATAC information
        data_manager.train_adata.obsm["X_latent"] = (
            vae_model.get_latent_representation()
        )

        dataset = data_manager.create_dif_dataset(
            data_manager.train_adata, vae_hash=vae_hash
        )

        train_cfg = dif_cfg.training
        dataloader = DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            num_workers=4,
            pin_memory=exp_cfg.pin_memory,
        )

        log_print("\nInitializing DiffusionForcing model...")
        model = DiffusionForcing(dif_cfg)
        checkpoint_dir = Path(exp_cfg.dif_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        max_steps = train_cfg.max_steps
        checkpoint_path = checkpoint_dir / f"checkpoint_step={max_steps}.ckpt"
        ckpt_path = None
        if checkpoint_path.exists():
            log_print(f"Found checkpoint at max_steps={max_steps}, skipping training.")
            log_print(f"Checkpoint: {checkpoint_path}")
            return
        else:
            existing_checkpoints = sorted(
                checkpoint_dir.glob("checkpoint_step=*.ckpt"),
                key=lambda p: int(p.stem.split("=")[1]),
            )
            if existing_checkpoints:
                ckpt_path = str(existing_checkpoints[-1])
                last_step = int(existing_checkpoints[-1].stem.split("=")[1])
                log_print(f"Resuming training from step {last_step}: {ckpt_path}")

        log_print(
            f"\nCreating Lightning trainer for {train_cfg.max_steps} steps..."
        )
        log_print(f"Output: {exp_cfg.dif_dir}")

        logger = WandbLogger(
            project=logging_cfg.wandb.project,
            name=f"{train_cfg.max_steps}steps",
            save_dir=exp_cfg.dif_dir,
            log_model=False,
            offline=(logging_cfg.wandb.mode == "offline"),
            config=dict(OmegaConf.to_container(config, resolve=True)),
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="checkpoint_{step}",
            save_top_k=-1,
            every_n_train_steps=train_cfg.checkpoint_every,
            save_last=True,
            save_on_train_epoch_end=False,
            verbose=True,
            enable_version_counter=False,
        )

        num_devices = exp_cfg.num_gpus
        strategy = exp_cfg.strategy
        log_print(
            f"\nStarting DiF training on {num_devices} GPU(s) (strategy: {strategy})..."
        )

        trainer = pl.Trainer(
            max_steps=train_cfg.max_steps,
            accelerator="gpu",
            devices=num_devices,
            strategy=strategy if num_devices > 1 else "auto",
            precision="16-mixed",
            gradient_clip_val=train_cfg.optimizer.gradient_clip_val,
            gradient_clip_algorithm="norm",
            log_every_n_steps=logging_cfg.wandb.log_frequency,
            default_root_dir=exp_cfg.dif_dir,
            logger=logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=True,
        )

        trainer.fit(model, train_dataloaders=dataloader, ckpt_path=ckpt_path)
        log_print("\nTraining complete!")
        log_print(f"Checkpoints saved in: {checkpoint_dir}/")

    else:
        raise ValueError(
            f"Invalid model type '{model_type}'. "
            f"Valid options: 'scvi', 'multivi', 'dif'"
        )


def generate(config: DictConfig | ListConfig, checkpoint_path: str | None = None) -> str:
    """Generate samples only. Returns output directory path."""
    data_cfg, exp_cfg, dif_cfg = (
        config.data,
        config.experiment,
        config.dif,
    )
    gen_cfg = dif_cfg.generation

    if not checkpoint_path:
        checkpoint_dir = Path(exp_cfg.dif_dir) / "checkpoints"
        max_steps = dif_cfg.training.max_steps
        checkpoint_path = checkpoint_dir / f"checkpoint_step={max_steps}.ckpt"
        if not checkpoint_path.exists():
            raise ValueError(
                f"Checkpoint not found for max_steps={max_steps}: {checkpoint_path}"
            )
        checkpoint_path = str(checkpoint_path)
        log_print(f"Using checkpoint for max_steps={max_steps}: {checkpoint_path}")

    mode = gen_cfg.mode
    base_dir = Path(exp_cfg.dif_dir)
    mode_abbrev = (
        "ns"
        if mode == "noise_start"
        else "ds" if mode == "data_start" else mode[:2]
    )

    checkpoint_name = Path(checkpoint_path).stem
    step_num = checkpoint_name.split("=")[1]
    infer_output_dir = base_dir / f"infer_{mode_abbrev}_step_{step_num}"
    infer_output_dir.mkdir(parents=True, exist_ok=True)

    log_path = infer_output_dir / "generate.log"
    setup_logging(log_path)
    log_print(f"Generation started at {log_path}")
    log_print(f"Mode: {mode}, Checkpoint: {checkpoint_path}")

    try:
        vae_type = exp_cfg.vae_to_use

        data_manager = DataManager(data_cfg, seed=exp_cfg.seed, vae_type=vae_type)
        inference = DiFInference(
            checkpoint_path, torch.device("cuda"), vae_dir=exp_cfg.vae_dir
        )

        log_print(f"Loading {vae_type.upper()} model from {exp_cfg.vae_dir}")

        vae_model = load_vae_model(vae_type, exp_cfg.vae_dir, data_manager, "train")

        data_manager.train_adata.obsm["X_latent"] = vae_model.get_latent_representation()

        if data_manager.test_adata is not None and len(data_manager.test_adata) > 0:
            if vae_type == "multivi":
                data_manager.setup_for_multivi(data_manager.test_mdata)
            else:
                data_manager.setup_for_scvi(data_manager.test_adata)

            vae_model_test = load_vae_model(vae_type, exp_cfg.vae_dir, data_manager, "test")
            data_manager.test_adata.obsm["X_latent"] = (
                vae_model_test.get_latent_representation()
            )

        log_print(f"\n{'='*60}")
        log_print(f"Generating {mode} samples")
        log_print(f"{'='*60}")
        inference_dir = inference.generate_and_save_samples(
            gen_config=GenerationConfig(
                mode=mode,
                num_samples=gen_cfg.n_samples,
                target_stages=None,
                batch_size=gen_cfg.batch_size,
            ),
            model_config=ModelConfig(
                uncertainty_scale=gen_cfg.uncertainty_scale,
                chunk_size=data_cfg.sequences.chunk_size,
                context_noise_scale=gen_cfg.context_noise_scale,
                scheduling_matrix=gen_cfg.scheduling_matrix,
            ),
            gen_data=GenerationData(
                data_manager=data_manager,
                vae_model=vae_model,
                output_dir=infer_output_dir,
            ),
        )
        log_print(f"\nGenerated samples saved to: {inference_dir}")

    finally:
        close_logging()

    return str(inference_dir)
