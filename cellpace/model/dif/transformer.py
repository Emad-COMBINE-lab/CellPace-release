"""Transformer denoiser with adaptive layer normalization for biological time."""

# Standard library
import math

# Third-party
import torch
import torch.nn as nn
from einops import rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, theta: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLNTransformerBlock(nn.Module):
    """Custom transformer block with Adaptive Layer Normalization (AdaLN).

    Replaces standard LayerNorm with AdaLN for biological time conditioning.
    Uses pre-norm architecture (modern best practice).
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Modern transformers use GELU

        # Layer norms (will be modulated by AdaLN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, attn_mask: torch.Tensor | None = None, is_causal: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input tensor (T, B, D)
            scale: Scale parameters for AdaLN (T, B, D)
            shift: Shift parameters for AdaLN (T, B, D)
            attn_mask: Optional attention mask
            is_causal: Whether to use causal attention
        """
        # Self-attention with pre-norm AdaLN
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale) + shift  # AdaLN modulation
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm, attn_mask=attn_mask, is_causal=is_causal
        )
        x = x + self.dropout1(attn_out)

        # Feed-forward with pre-norm AdaLN
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale) + shift  # AdaLN modulation
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + self.dropout2(ff_out)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        x_dim,
        external_cond_dim=0,
        size=128,
        num_layers=4,
        nhead=4,
        dim_feedforward=512,
        dropout=0.0,
        seq_len=6,
        gap_config=None,
    ):
        super(Transformer, self).__init__()
        self.external_cond_dim = external_cond_dim
        self.seq_len = seq_len

        # Simplified gap configuration
        assert (
            gap_config is not None
        ), "gap_config must be provided in architecture config"
        self.gap_enabled = gap_config["enabled"]
        self.gap_scale = gap_config["gap_scale"]

        # Use custom transformer blocks with AdaLN instead of standard PyTorch transformer
        self.num_layers = num_layers
        self.transformer_blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(size, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        k_embed_dim = size // 2

        # Biological time embedding - single MLP for normalized position + gap
        # Always requires gap information for proper temporal modeling
        assert self.gap_enabled, "Gap modeling must be enabled for temporal sequences"

        # Accepts 2D input: [normalized_position, gap_feature]
        self.biological_time_mlp = nn.Sequential(
            nn.Linear(2, size // 2),  # Back to 2D input
            nn.SiLU(),
            nn.Linear(size // 2, size * 2),  # Output 2x size for AdaLN scale and shift
        )

        # AdaLN-Zero initialization: Start with identity mapping
        # This improves training stability and prevents early mode collapse
        # Following PixArt-α and Stable Diffusion 3
        nn.init.zeros_(self.biological_time_mlp[-1].weight)
        nn.init.zeros_(self.biological_time_mlp[-1].bias)

        self.k_embed = SinusoidalPosEmb(dim=k_embed_dim)

        # MLP processes latent + noise (+ external conditions) - NO TIME (following DiF-main)
        self.init_mlp = nn.Sequential(
            nn.Linear(x_dim + k_embed_dim + external_cond_dim, size),
            nn.ReLU(),
            nn.Linear(size, size),
        )
        self.out = nn.Linear(size, x_dim)

    def forward(self, x: torch.Tensor, k: int, stage_positions: torch.Tensor, external_cond: torch.Tensor | None = None, is_causal: bool = False) -> torch.Tensor:
        # x.shape (T, B, C)
        # k.shape (T, B) - noise levels
        # stage_positions.shape (T, B, 2) - [normalized position, time gap]
        # optional external_cond.shape (T, B, C) - future: cell types, etc.

        seq_len, batch_size, _ = x.shape

        # stage positions are required with gap information
        assert stage_positions is not None, "stage_positions must be provided"
        assert (
            stage_positions.dim() == 3 and stage_positions.shape[2] == 2
        ), f"stage_positions must be (T, B, 2) with [position, gap], got {stage_positions.shape}"

        # Extract normalized positions and gaps
        normalized_positions = stage_positions[..., 0]  # (T, B)
        gaps = stage_positions[..., 1]  # (T, B)

        # Linear scaling for gaps
        gap_features = gaps * self.gap_scale

        # Combine position and gap for biological time embedding (back to 2D)
        bio_input = torch.stack(
            [normalized_positions, gap_features], dim=-1
        )  # (T, B, 2)
        biological_params = self.biological_time_mlp(bio_input)
        scale, shift = biological_params.chunk(
            2, dim=-1
        )  # Split into scale and shift for AdaLN

        # Get noise embedding
        k_embed = rearrange(self.k_embed(k.flatten()), "(t b) d -> t b d", t=seq_len)

        # Following DiF-main pattern: concatenate data features (NOT time)
        x = torch.cat((x, k_embed), dim=-1)
        if external_cond is not None:
            x = torch.cat((x, external_cond), dim=-1)
        elif self.external_cond_dim > 0:
            raise ValueError(
                f"Model configured with external_cond_dim={self.external_cond_dim} "
                f"but no external conditions provided. Set external_cond_dim=0 in config."
            )

        # MLP processes latent + noise (+ external conditions) - NO TIME!
        x = self.init_mlp(x)

        # Apply custom transformer blocks with integrated AdaLN
        mask = (
            nn.Transformer.generate_square_subsequent_mask(len(x), x.device)
            if is_causal
            else None
        )

        # Process through custom transformer blocks with biological time modulation
        for block in self.transformer_blocks:
            x = block(x, scale, shift, attn_mask=mask, is_causal=is_causal)

        x = self.out(x)

        return x
