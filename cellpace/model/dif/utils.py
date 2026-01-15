"""Noise schedules, config dataclasses, and tensor utilities for DiF."""

# Standard library
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Third-party
import torch


@dataclass
class GenerationConfig:
    """Configuration for sample generation."""

    mode: str  # 'noise_start' or 'data_start'
    num_samples: int  # Number of samples per stage
    target_stages: list[str | None] = None  # None means all stages
    batch_size: int = 200  # Batch size for memory efficiency
    decode_batch_size: int = 512  # Batch size for VAE decoding
    verbose: bool = True  # Whether to print progress


@dataclass
class ModelConfig:
    """Configuration for model generation parameters."""

    uncertainty_scale: float  # Controls temporal uncertainty propagation
    chunk_size: int  # Number of positions to generate at once
    context_noise_scale: float = (
        0.1  # From config - noise scale for context during generation
    )
    scheduling_matrix: str = (
        "pyramid"  # Scheduling type: pyramid, full_sequence, autoregressive, trapezoid
    )


@dataclass
class GenerationData:
    """Data required for generation."""

    data_manager: Any  # DataManager instance
    vae_model: object  # VAE model (scVI or MultiVI) for decoding
    output_dir: Path  # Where to save generated samples


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    f, b = t.shape
    out = a[t]
    return out.reshape(f, b, *((1,) * (len(x_shape) - 2)))


def linear_beta_schedule(
    timesteps: int, scale_factor: int = 1000, beta_start_scale: float = 0.0001, beta_end_scale: float = 0.02
) -> torch.Tensor:
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = scale_factor / timesteps
    beta_start = scale * beta_start_scale
    beta_end = scale * beta_end_scale
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps: int, start: float = -3, end: float = 3, tau: float = 1, clamp_min: float = 1e-5) -> torch.Tensor:
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
