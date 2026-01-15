"""Diffusion process with per-position noise scheduling and DDIM sampling."""

# Standard library
from collections import namedtuple
from typing import Any, Callable

# Third-party
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

# Local
from .transformer import Transformer
from .utils import (
    cosine_beta_schedule,
    extract,
    linear_beta_schedule,
    sigmoid_beta_schedule,
)

ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_noise", "pred_x_start", "model_out"]
)


class Diffusion(nn.Module):
    # Special thanks to lucidrains for the implementation of the base Diffusion model
    # https://github.com/lucidrains/denoising-diffusion-pytorch

    def __init__(
        self,
        x_shape: torch.Size,
        external_cond_dim: int,
        is_causal: bool,
        diffusion_cfg,
        transformer_cfg,
        gap_config,
        seq_len: int,
        snr_clip: float = 5.0,
    ):
        super().__init__()

        # Use provided config sections directly
        self.diffusion_cfg = diffusion_cfg
        self.arch = transformer_cfg
        self.schedule_cfg = diffusion_cfg.schedule
        self.sampling_cfg = diffusion_cfg.sampling
        self.gap_config = gap_config
        self.seq_len = seq_len

        # Direct parameters
        self.x_shape = x_shape
        self.external_cond_dim = external_cond_dim
        self.is_causal = is_causal

        # Loss weighting
        self.snr_clip = snr_clip

        # Store noise clipping value
        self.clip_noise = self.diffusion_cfg.noise_clip

        self._build_model()
        self._build_buffer()

    def _build_model(self) -> None:
        x_channel = self.x_shape[0]
        if len(self.x_shape) == 1:
            self.model = Transformer(
                x_dim=x_channel,
                external_cond_dim=self.external_cond_dim,
                size=self.arch.hidden_dim,
                num_layers=self.arch.n_layers,
                nhead=self.arch.n_heads,
                dim_feedforward=self.arch.feedforward_dim,
                seq_len=self.seq_len,
                gap_config=self.gap_config,
                dropout=self.arch.dropout,
            )
        else:
            raise ValueError(f"unsupported input shape {self.x_shape}")

    def _build_buffer(self) -> None:
        schedule_type = self.schedule_cfg.type

        if schedule_type == "linear":
            betas = linear_beta_schedule(
                self.diffusion_cfg.timesteps,
                scale_factor=self.schedule_cfg.linear.scale_factor,
                beta_start_scale=self.schedule_cfg.linear.beta_start_scale,
                beta_end_scale=self.schedule_cfg.linear.beta_end_scale,
            )
        elif schedule_type == "cosine":
            # Get cosine parameters from config
            betas = cosine_beta_schedule(
                self.diffusion_cfg.timesteps, s=self.schedule_cfg.cosine.offset
            )
        elif schedule_type == "sigmoid":
            # Get sigmoid parameters from config
            betas = sigmoid_beta_schedule(
                self.diffusion_cfg.timesteps,
                start=self.schedule_cfg.sigmoid.start,
                end=self.schedule_cfg.sigmoid.end,
                tau=self.schedule_cfg.sigmoid.tau,
                clamp_min=self.schedule_cfg.sigmoid.clamp_min,
            )
        else:
            raise ValueError(f"unknown beta schedule {schedule_type}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # sampling related parameters
        assert self.diffusion_cfg.sampling_timesteps <= self.diffusion_cfg.timesteps
        self.is_ddim_sampling = (
            self.diffusion_cfg.sampling_timesteps < self.diffusion_cfg.timesteps
        )

        # helper function to register buffer from float64 to float32
        def register_buffer(name, val):
            self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer("posterior_variance", posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # derive loss weight
        # https://arxiv.org/abs/2303.09556
        # snr: signal noise ratio
        # Add small epsilon to avoid division by zero in edge cases
        snr = alphas_cumprod / (1 - alphas_cumprod).clamp(min=1e-8)
        clipped_snr = snr.clone()
        clipped_snr.clamp_(max=self.snr_clip)

        register_buffer("clipped_snr", clipped_snr)
        register_buffer("snr", snr)

    def add_shape_channels(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, f"... -> ...{' 1' * len(self.x_shape)}")

    def model_predictions(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        stage_positions: torch.Tensor,
        external_cond: torch.Tensor | None = None,
    ) -> Any:
        model_output = self.model(
            x, t, stage_positions, external_cond, is_causal=self.is_causal
        )

        if self.diffusion_cfg.objective == "pred_noise":
            pred_noise = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        elif self.diffusion_cfg.objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.diffusion_cfg.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        else:
            # Default to pred_noise
            pred_noise = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        return ModelPrediction(pred_noise, x_start, model_output)

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(
        self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_mean_variance(
        self, x_start: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
            noise = torch.clamp(
                noise, -self.diffusion_cfg.noise_clip, self.diffusion_cfg.noise_clip
            )

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        stage_positions: torch.Tensor,
        external_cond: torch.Tensor | None = None,
    ) -> tuple:
        model_pred = self.model_predictions(
            x=x, t=t, stage_positions=stage_positions, external_cond=external_cond
        )
        x_start = model_pred.pred_x_start
        return self.q_posterior(x_start=x_start, x_t=x, t=t)

    def compute_loss_weights(
        self, noise_levels: torch.Tensor, time_gaps: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute Min-SNR loss weights for the given noise levels.

        Based on https://arxiv.org/abs/2303.09556
        """
        snr = self.snr[noise_levels]
        clipped_snr = self.clipped_snr[noise_levels]

        match self.diffusion_cfg.objective:
            case "pred_noise":
                base_weights = clipped_snr / snr.clamp(min=1e-8)
            case "pred_x0":
                base_weights = clipped_snr
            case "pred_v":
                base_weights = clipped_snr / (snr + 1).clamp(min=1e-8)
            case _:
                raise ValueError(
                    f"unknown objective {self.diffusion_cfg.objective}"
                )

        return base_weights

    def forward(
        self,
        x: torch.Tensor,
        stage_positions: torch.Tensor,
        external_cond: torch.Tensor | None,
        noise_levels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract gaps if provided (stage_positions includes both position and gap)
        time_gaps = None
        if stage_positions.dim() == 3 and stage_positions.shape[2] == 2:
            time_gaps = stage_positions[..., 1]  # Extract gap component

        noise = torch.randn_like(x)
        noise = torch.clamp(
            noise, -self.diffusion_cfg.noise_clip, self.diffusion_cfg.noise_clip
        )

        noised_x = self.q_sample(x_start=x, t=noise_levels, noise=noise)
        model_pred = self.model_predictions(
            x=noised_x,
            t=noise_levels,
            stage_positions=stage_positions,
            external_cond=external_cond,
        )

        pred = model_pred.model_out
        x_pred = model_pred.pred_x_start

        if self.diffusion_cfg.objective == "pred_noise":
            target = noise
        elif self.diffusion_cfg.objective == "pred_x0":
            target = x
        elif self.diffusion_cfg.objective == "pred_v":
            target = self.predict_v(x, noise_levels, noise)
        else:
            raise ValueError(f"unknown objective {self.diffusion_cfg.objective}")

        loss = F.mse_loss(pred, target.detach(), reduction="none")
        loss_weight = self.compute_loss_weights(noise_levels, time_gaps)
        loss_weight = loss_weight.view(*loss_weight.shape, *((1,) * (loss.ndim - 2)))
        loss = loss * loss_weight

        return x_pred, loss

    def sample_step(
        self,
        x: torch.Tensor,
        stage_positions: torch.Tensor,
        external_cond: torch.Tensor | None,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        guidance_fn: Callable | None = None,
    ) -> torch.Tensor:
        real_steps = torch.linspace(
            -1,
            self.diffusion_cfg.timesteps - 1,
            steps=self.diffusion_cfg.sampling_timesteps + 1,
            device=x.device,
        ).long()

        # convert noise levels (0 ~ sampling_timesteps) to real noise levels (-1 ~ timesteps - 1)
        curr_noise_level = real_steps[curr_noise_level]
        next_noise_level = real_steps[next_noise_level]

        if self.is_ddim_sampling:
            return self.ddim_sample_step(
                x=x,
                stage_positions=stage_positions,
                external_cond=external_cond,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                guidance_fn=guidance_fn,
            )

        # Validate noise levels for DDPM sampling
        assert torch.all(
            (curr_noise_level - 1 == next_noise_level)
            | ((curr_noise_level == -1) & (next_noise_level == -1))
        ), "Wrong noise level given for ddpm sampling."

        assert (
            self.diffusion_cfg.sampling_timesteps == self.diffusion_cfg.timesteps
        ), "sampling_timesteps should be equal to timesteps for ddpm sampling."

        return self.ddpm_sample_step(
            x=x,
            stage_positions=stage_positions,
            external_cond=external_cond,
            curr_noise_level=curr_noise_level,
            guidance_fn=guidance_fn,
        )

    def ddpm_sample_step(
        self,
        x: torch.Tensor,
        stage_positions: torch.Tensor,
        external_cond: torch.Tensor | None,
        curr_noise_level: torch.Tensor,
        guidance_fn: Callable | None = None,
    ) -> torch.Tensor:
        clipped_curr_noise_level = torch.where(
            curr_noise_level < 0,
            torch.full_like(
                curr_noise_level,
                self.diffusion_cfg.stabilization_level - 1,
                dtype=torch.long,
            ),
            curr_noise_level,
        )

        # treating as stabilization would require us to scale with sqrt of alpha_cum
        orig_x = x.clone().detach()
        scaled_context = self.q_sample(x, clipped_curr_noise_level)
        x = torch.where(
            self.add_shape_channels(curr_noise_level < 0), scaled_context, orig_x
        )

        if guidance_fn is not None:
            raise ValueError(
                "Guidance function not supported. Set guidance_fn=None."
            )

        else:
            model_mean, _, model_log_variance = self.p_mean_variance(
                x=x,
                t=clipped_curr_noise_level,
                stage_positions=stage_positions,
                external_cond=external_cond,
            )

        noise = torch.where(
            self.add_shape_channels(clipped_curr_noise_level > 0),
            torch.randn_like(x),
            0,
        )
        noise = torch.clamp(
            noise, -self.diffusion_cfg.noise_clip, self.diffusion_cfg.noise_clip
        )
        x_pred = model_mean + torch.exp(0.5 * model_log_variance) * noise

        # only update frames where the noise level decreases
        return torch.where(
            self.add_shape_channels(curr_noise_level == -1), orig_x, x_pred
        )

    def ddim_sample_step(
        self,
        x: torch.Tensor,
        stage_positions: torch.Tensor,
        external_cond: torch.Tensor | None,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        guidance_fn: Callable | None = None,
    ) -> torch.Tensor:
        # convert noise level -1 to self.stabilization_level - 1
        clipped_curr_noise_level = torch.where(
            curr_noise_level < 0,
            torch.full_like(
                curr_noise_level,
                self.diffusion_cfg.stabilization_level - 1,
                dtype=torch.long,
            ),
            curr_noise_level,
        )

        # treating as stabilization would require us to scale with sqrt of alpha_cum
        orig_x = x.clone().detach()
        scaled_context = self.q_sample(
            x,
            clipped_curr_noise_level,
            noise=torch.zeros_like(x),
        )
        x = torch.where(
            self.add_shape_channels(curr_noise_level < 0), scaled_context, orig_x
        )

        alpha = self.alphas_cumprod[clipped_curr_noise_level]
        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level),
            self.alphas_cumprod[next_noise_level],
        )
        sigma = torch.where(
            next_noise_level < 0,
            torch.zeros_like(next_noise_level),
            self.sampling_cfg.eta
            * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
        )
        c = (1 - alpha_next - sigma**2).sqrt()

        alpha_next = self.add_shape_channels(alpha_next)
        c = self.add_shape_channels(c)
        sigma = self.add_shape_channels(sigma)

        if guidance_fn is not None:
            with torch.enable_grad():
                x = x.detach().requires_grad_()

                model_pred = self.model_predictions(
                    x=x,
                    t=clipped_curr_noise_level,
                    stage_positions=stage_positions,
                    external_cond=external_cond,
                )

                guidance_loss = guidance_fn(model_pred.pred_x_start)
                grad = -torch.autograd.grad(
                    guidance_loss,
                    x,
                )[0]

                pred_noise = model_pred.pred_noise + (1 - alpha_next).sqrt() * grad
                x_start = self.predict_start_from_noise(
                    x, clipped_curr_noise_level, pred_noise
                )

        else:
            model_pred = self.model_predictions(
                x=x,
                t=clipped_curr_noise_level,
                stage_positions=stage_positions,
                external_cond=external_cond,
            )
            x_start = model_pred.pred_x_start
            pred_noise = model_pred.pred_noise

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise

        # only update frames where the noise level decreases
        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(
            self.add_shape_channels(mask),
            orig_x,
            x_pred,
        )

        return x_pred
