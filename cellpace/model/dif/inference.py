"""Inference engine for generating temporal single-cell trajectories."""

# Standard library
import pickle
from pathlib import Path
from typing import Any

# Third-party
import anndata as ad
import numpy as np
import pandas as pd
import torch

from ...utils.logging import log_print

# Local
from .core import DiffusionForcing
from .utils import GenerationConfig, GenerationData, ModelConfig


class DiFInference:
    """Handles DiF model inference with various scheduling strategies."""

    def __init__(self, checkpoint_path: str, device: torch.device, vae_dir: str = None, enable_gradients: bool = False):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.enable_gradients = enable_gradients  # Allow gradient flow for training LoRA adapters

        # Load PyTorch Lightning checkpoint
        log_print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        # Extract iteration number from Lightning's global_step
        self.iteration = checkpoint.get("global_step", None)
        if self.iteration is not None:
            log_print(f"Checkpoint iteration: {self.iteration}")

        # Load model using Lightning's load_from_checkpoint
        self.model = DiffusionForcing.load_from_checkpoint(
            checkpoint_path, map_location=device
        )

        # Weights already loaded via load_from_checkpoint
        log_print("Loading model weights for inference")

        # Store both the full model and the diffusion component
        self.full_model = self.model.to(device)
        self.full_model.eval()
        self.model = self.full_model.diffusion_model

        # Load metadata from VAE directory (shared metadata location)
        vae_dir = Path(vae_dir)
        metadata_path = vae_dir / "metadata.pkl"
        log_print(f"Loading shared metadata from {metadata_path}")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Extract required fields - fail if any are missing
        required_fields = ["stage_min", "stage_max"]
        for field in required_fields:
            if field not in metadata:
                raise KeyError(f"Required field '{field}' not found in metadata.pkl")

        self.data_stats = {
            "latent_dim": self.full_model.arch_cfg.input_dim,
            "stage_min": metadata["stage_min"],
            "stage_max": metadata["stage_max"],
        }

        # Load per-stage library sizes - handle both scVI and MultiVI formats
        if "stage_to_lib_sizes" in metadata:
            self.stage_to_lib_sizes = metadata["stage_to_lib_sizes"]
            self.stage_to_atac_lib_sizes = None  # scVI doesn't have ATAC
            log_print(f"Loaded metadata - stage range: [{metadata['stage_min']}, {metadata['stage_max']}]")
            log_print(f"Loaded per-stage library sizes for {len(self.stage_to_lib_sizes)} stages (scVI)")
        elif "stage_to_rna_lib_sizes" in metadata:
            self.stage_to_lib_sizes = metadata["stage_to_rna_lib_sizes"]
            self.stage_to_atac_lib_sizes = metadata["stage_to_atac_lib_sizes"]
            log_print(f"Loaded metadata - stage range: [{metadata['stage_min']}, {metadata['stage_max']}]")
            log_print(f"Loaded per-stage RNA library sizes for {len(self.stage_to_lib_sizes)} stages (MultiVI)")
            log_print(f"Loaded per-stage ATAC library sizes for {len(self.stage_to_atac_lib_sizes)} stages (MultiVI)")
        else:
            raise KeyError(
                "metadata.pkl must contain either 'stage_to_lib_sizes' (scVI) or "
                "'stage_to_rna_lib_sizes' (MultiVI)"
            )

        # Store configuration from the loaded model
        self.args: Any = type(
            "Args",
            (),
            {
                "seq_len": self.full_model.arch_cfg.sequence_length,
                "loss_weighting": self.full_model.training_cfg.loss.strategy,
            },
        )()

        self.model_cfg: Any = type(
            "ModelConfig",
            (),
            {
                "stabilization_level": self.full_model.diffusion_cfg.stabilization_level,
                "clip_noise": self.full_model.diffusion_cfg.noise_clip,
            },
        )()

        # Set sampling parameters
        self.model.sampling_timesteps = self.full_model.diffusion_cfg.sampling_timesteps
        self.model.timesteps = self.full_model.diffusion_cfg.timesteps

        log_print(f"Loaded DiF model v3 from {checkpoint_path}")
        log_print(
            f"Model config: seq_len={self.args.seq_len}, timesteps={self.model.timesteps}, sampling_timesteps={self.model.sampling_timesteps}"
        )

    def generate_future_cells(
        self,
        context_cells: torch.Tensor | None,
        context_stages: list[Any | None],  # Can be int or str stage labels
        target_stages: list[Any],  # Can be int or str stage labels
        num_samples: int,
        uncertainty_scale: float,
        stabilize_context: bool,
        context_noise_scale: float,
        verbose: bool,
        chunk_size: int,
        scheduling_type: str,
    ) -> dict[str, torch.Tensor]:
        """
        Generate future cells using specified scheduling with adjustable temporal coupling.

        Unified interface that handles both pure noise start (empty context) and
        context-based generation, following the DiF paper's philosophy that all
        generation is autoregressive.

        Args:
            context_cells: [N, D] real cells from context stages (None for pure noise start)
            context_stages: list of context stage values (None for pure noise start)
            target_stages: list of target stage values to generate
            num_samples: Number of samples to generate per target stage
            uncertainty_scale: Controls temporal uncertainty propagation:
                              - < 1.0: Tighter coupling (more deterministic development)
                              - = 1.0: Standard development (default)
                              - > 1.0: Looser coupling (more stochastic development)
                              - = 50.0: Strict autoregressive (cell fate decisions)
                              Examples:
                              - 0.5: Smooth developmental transitions (gradual differentiation)
                              - 1.0: Normal embryonic development
                              - 2.0: Highly variable development (stress conditions)
                              - 50.0: Strict developmental checkpoints
            stabilize_context: Add small noise to context for stability
            context_noise_scale: Scale of noise for context stabilization
            verbose: Print progress
            chunk_size: If > 0, use sliding window generation
            scheduling_type: Type of scheduling matrix to use:
                           - "pyramid": Pyramid scheduling (default, best for development)
                           - "full_sequence": Full sequence denoising
                           - "autoregressive": Strict autoregressive (like pyramid with max uncertainty)
                           - "trapezoid": Trapezoid scheduling

        Returns:
            Dictionary with generated cells for each target stage
        """
        device = self.device
        latent_dim = self.data_stats["latent_dim"]

        # Handle unified context: empty context means pure noise start
        if context_cells is None or context_stages is None:
            # Pure noise start
            context_len = 0
            context_stages = []
            context_cells = torch.empty(
                0, num_samples, latent_dim, device=device
            )  # Empty tensor [T=0, B, D]
        else:
            context_len = len(context_stages)
            # Ensure context_cells is on the right device
            if context_cells.device != device:
                context_cells = context_cells.to(device)

        target_len = len(target_stages)
        total_len = context_len + target_len

        # Normalize stage positions
        all_stages = context_stages + target_stages

        # Extract numeric values from stage strings
        # Support both formats for convenience: "stage_12" or 12
        numeric_stages = []
        for s in all_stages:
            if isinstance(s, str) and "_" in s:
                # Handle any 'prefix_XX' format (stage_10, hpf_48, etc.)
                num = int(s.split("_")[1])
            else:
                num = int(s)
            numeric_stages.append(num)

        # Use the normalization parameters from training
        stage_min = self.data_stats["stage_min"]
        stage_max = self.data_stats["stage_max"]

        # Normalize using training range
        normalized_stages = [
            (s - stage_min) / (stage_max - stage_min) for s in numeric_stages
        ]

        # Compute time gaps between consecutive stages
        time_gaps = [0.0]  # First position has no gap
        for i in range(1, len(numeric_stages)):
            gap = numeric_stages[i] - numeric_stages[i - 1]
            time_gaps.append(float(gap))

        # Generate scheduling matrix for debug output only
        # The actual schedule used is created in _generate_with_sliding_window
        if verbose:
            # Generate a preview schedule for debugging
            preview_schedule = self.generate_scheduling_matrix(
                horizon=target_len,
                scheduling_type=scheduling_type,
                uncertainty_scale=uncertainty_scale,
            )
            # Add context zeros if needed
            if context_len > 0:
                context_zeros = np.zeros(
                    (preview_schedule.shape[0], context_len), dtype=np.int64
                )
                schedule = np.concatenate([context_zeros, preview_schedule], axis=1)
            else:
                schedule = preview_schedule

            log_print("\n=== Generation Setup Debug ===")
            if context_len > 0:
                log_print(
                    f"Context stages: {context_stages} (positions 0-{context_len-1})"
                )
            else:
                log_print("Context stages: None (pure noise generation)")
            log_print(
                f"Target stages: {target_stages} (positions {context_len}-{total_len-1})"
            )
            log_print(
                f"Schedule type: {scheduling_type} (uncertainty_scale={uncertainty_scale})"
            )
            log_print(
                f"Schedule shape: {schedule.shape} (height={schedule.shape[0]}, positions={schedule.shape[1]})"
            )
            log_print(f"Schedule first step: {schedule[0]}")
            log_print(f"Schedule last step: {schedule[-1]}")

            # More detailed schedule inspection
            if context_len > 0:
                log_print(
                    f"Context positions noise levels: {schedule[0, :context_len]} (should be all 0s)"
                )
            log_print(f"Target positions start with noise: {schedule[0, context_len:]}")

            # Describe the scheduling pattern
            if scheduling_type == "pyramid":
                log_print("\nPyramid pattern: Later positions start denoising later")
            elif scheduling_type == "full_sequence":
                log_print("\nFull sequence: All positions denoised uniformly")
            elif scheduling_type == "autoregressive":
                log_print(
                    "\nAutoregressive: Strict position-by-position (max uncertainty)"
                )
            elif scheduling_type == "trapezoid":
                log_print("\nTrapezoid: Symmetric denoising from edges to center")

            log_print("\nFirst 10 denoising steps:")
            for i in range(min(10, schedule.shape[0])):
                if total_len <= 10:
                    log_print(f"  Step {i}: {schedule[i]}")
                else:
                    # Show context and first few target positions
                    if context_len > 0:
                        log_print(
                            f"  Step {i}: ctx={schedule[i, :context_len]} | tgt={schedule[i, context_len:context_len+5]}..."
                        )
                    else:
                        log_print(f"  Step {i}: {schedule[i, :min(8, total_len)]}...")

            # Verify monotonicity (noise should decrease or stay same)
            violations = []
            for pos in range(total_len):
                for step in range(1, min(10, schedule.shape[0])):
                    if schedule[step, pos] > schedule[step - 1, pos]:
                        violations.append(
                            f"pos {pos}: {schedule[step-1, pos]}→{schedule[step, pos]} at step {step}"
                        )

            if violations:
                log_print("\nWarning: MONOTONICITY VIOLATIONS (noise should decrease):")
                for v in violations[:3]:
                    log_print(f"  {v}")
                if len(violations) > 3:
                    log_print(f"  ... and {len(violations)-3} more")
            else:
                log_print("\n  [green]✓[/green] Schedule is monotonic (noise properly decreases)")

            log_print("===========================")

        # Initialize sequences with [T, B, D] format for training consistency
        # Start with context cells, then concatenate noise chunks during generation

        # Fill context with real cells (no stabilization at initialization)
        if context_len > 0:
            # Context must be structured: [context_len, num_samples, latent_dim]
            x_seq = context_cells[:context_len, :].clone()
        else:
            # Pure noise start - initialize empty tensor
            x_seq = torch.empty(0, num_samples, latent_dim, device=device)

        # Noise chunks will be concatenated during the generation loop

        # Initially prepare stage positions and time gaps for context only
        # Will extend as we generate chunks
        if context_len > 0:
            stage_positions = (
                torch.tensor(
                    normalized_stages[:context_len], dtype=torch.float32, device=device
                )
                .unsqueeze(1)
                .expand(-1, num_samples)
            )

            time_gaps_tensor = (
                torch.tensor(
                    time_gaps[:context_len], dtype=torch.float32, device=device
                )
                .unsqueeze(1)
                .expand(-1, num_samples)
            )
        else:
            stage_positions = torch.empty(0, num_samples, device=device)
            time_gaps_tensor = torch.empty(0, num_samples, device=device)

        # Determine chunk size for generation
        # If chunk_size is 0 or larger than target_len, generate all at once
        effective_chunk_size = chunk_size if chunk_size > 0 else target_len

        # Always use sliding window approach (with chunk_size = target_len for single-shot)
        return self._generate_with_sliding_window(
            x_seq,
            stage_positions,
            time_gaps_tensor,
            normalized_stages,
            time_gaps,
            context_len,
            target_len,
            target_stages,
            context_stages,
            num_samples,
            latent_dim,
            device,
            effective_chunk_size,
            uncertainty_scale,
            verbose,
            scheduling_type,
        )

    def _generate_with_sliding_window(
        self,
        x_seq,
        stage_positions,
        time_gaps_tensor,
        normalized_stages,
        time_gaps,
        context_len,
        target_len,
        target_stages,
        context_stages,
        num_samples,
        latent_dim,
        device,
        chunk_size,
        uncertainty_scale,
        verbose,
        scheduling_type,
    ):
        """
        Generate long sequences using sliding window approach from DiF paper.

        This method generates chunks of future positions, keeps the generated
        chunks as context, and slides forward to generate the next chunk.

        Args:
            scheduling_type: Type of scheduling matrix ("pyramid", "full_sequence",
                           "autoregressive", "trapezoid")
        """
        results = {}
        curr_pos = context_len  # Current position in the sequence

        # Start with context, will concatenate noise chunks (match df_base.py)
        generated_seq = x_seq.clone()

        # Get clip_noise value for clamping
        clip_noise = self.full_model.diffusion_cfg.noise_clip

        while curr_pos < context_len + target_len:
            # Determine horizon for this chunk
            # OPTIMIZATION: For the first step with no context, generate seq_len positions
            # to match the training distribution (avoids out-of-distribution warm-up phase)
            if curr_pos == 0 and context_len == 0:
                # First step: generate full seq_len to match training distribution
                horizon = min(self.args.seq_len, target_len)
            else:
                # Subsequent steps: use normal chunk_size
                horizon = min(chunk_size, context_len + target_len - curr_pos)

            # Create and concatenate new noise chunk
            chunk = torch.randn(horizon, num_samples, latent_dim, device=device)
            chunk = torch.clamp(chunk, -clip_noise, clip_noise)
            generated_seq = torch.cat([generated_seq, chunk], 0)

            # Also extend stage positions and gaps for the new chunk
            chunk_stages = (
                torch.tensor(
                    normalized_stages[curr_pos : curr_pos + horizon],
                    dtype=torch.float32,
                    device=device,
                )
                .unsqueeze(1)
                .expand(-1, num_samples)
            )
            stage_positions = torch.cat([stage_positions, chunk_stages], 0)

            chunk_gaps = (
                torch.tensor(
                    time_gaps[curr_pos : curr_pos + horizon],
                    dtype=torch.float32,
                    device=device,
                )
                .unsqueeze(1)
                .expand(-1, num_samples)
            )
            time_gaps_tensor = torch.cat([time_gaps_tensor, chunk_gaps], 0)

            # Verbose output showing generation steps
            if verbose:
                # For sliding window, we need to know which positions we're actually using
                max_context = self.args.seq_len - horizon

                # Determine actual window context
                if curr_pos <= max_context:
                    window_start = 0
                    context_positions = list(range(curr_pos))
                else:
                    window_start = curr_pos - max_context
                    context_positions = list(range(window_start, curr_pos))

                # Map positions to stage names for context
                context_stage_names = []
                for pos in context_positions:
                    if context_stages and pos < len(context_stages):
                        context_stage_names.append(context_stages[pos])
                    else:
                        # This is a previously generated position or pure noise
                        if context_stages:
                            gen_idx = pos - len(context_stages)
                        else:
                            gen_idx = pos  # Pure noise: all are generated
                        if gen_idx < len(target_stages):
                            context_stage_names.append(target_stages[gen_idx])

                # Map target positions
                target_positions = list(range(curr_pos, curr_pos + horizon))
                target_stage_names = []
                for pos in target_positions:
                    if context_stages:
                        gen_idx = pos - len(context_stages)
                    else:
                        gen_idx = pos  # Pure noise: all are generated
                    if gen_idx < len(target_stages):
                        target_stage_names.append(target_stages[gen_idx])

                # Show the actual horizon being generated, not just based on chunk_size
                if horizon == 1:
                    log_print(
                        f"  Step {curr_pos - context_len + 1}: {context_stage_names} → {target_stage_names[0]}"
                    )
                else:
                    log_print(
                        f"  Step {curr_pos - context_len + 1}: {context_stage_names} → {target_stage_names}"
                    )

            # CRITICAL: Implement sliding window to maintain fixed sequence length
            # Model was trained on seq_len, so we must maintain that window size

            # Generate scheduling matrix for target positions only (following dif-main)
            target_schedule = self.generate_scheduling_matrix(
                horizon=horizon,
                scheduling_type=scheduling_type,
                uncertainty_scale=uncertainty_scale,
            )

            # Denoise this chunk
            num_denoising_steps = target_schedule.shape[0]

            # Determine sliding window start
            start_frame = max(0, curr_pos + horizon - self.args.seq_len)

            for step in range(
                num_denoising_steps - 1
            ):  # -1 because we need pairs (from, to)
                # Build noise levels
                # Concatenate zeros for previous positions + schedule for current chunk
                from_noise_levels = np.concatenate(
                    (np.zeros((curr_pos,), dtype=np.int64), target_schedule[step])
                )[:, None].repeat(num_samples, axis=1)

                to_noise_levels = np.concatenate(
                    (np.zeros((curr_pos,), dtype=np.int64), target_schedule[step + 1])
                )[:, None].repeat(num_samples, axis=1)

                from_noise_levels = torch.from_numpy(from_noise_levels).to(device)
                to_noise_levels = torch.from_numpy(to_noise_levels).to(device)

                # Extract sliding window for stage positions and gaps
                window_stage_positions = stage_positions[
                    start_frame : curr_pos + horizon, :
                ]
                window_time_gaps = time_gaps_tensor[start_frame : curr_pos + horizon, :]

                # Stack temporal features
                temporal_features = torch.stack(
                    [window_stage_positions, window_time_gaps], dim=-1
                )  # [T, B, 2]

                # Update generated sequence by sampling
                # Conditionally disable gradients based on enable_gradients flag
                if self.enable_gradients:
                    # Allow gradients for training (e.g., LoRA fine-tuning)
                    generated_seq[start_frame:] = self.model.sample_step(
                        x=generated_seq[start_frame:],
                        stage_positions=temporal_features,
                        external_cond=None,
                        curr_noise_level=from_noise_levels[start_frame:],
                        next_noise_level=to_noise_levels[start_frame:],
                        guidance_fn=None,
                    )
                else:
                    # Standard inference mode - disable gradients for memory efficiency
                    with torch.no_grad():
                        generated_seq[start_frame:] = self.model.sample_step(
                            x=generated_seq[start_frame:],
                            stage_positions=temporal_features,
                            external_cond=None,
                            curr_noise_level=from_noise_levels[start_frame:],
                            next_noise_level=to_noise_levels[start_frame:],
                            guidance_fn=None,
                        )

            # Move forward
            curr_pos += horizon

        # Extract results for each target stage - use original stage as key
        for i, stage in enumerate(target_stages):
            # Use the original stage format as the key (could be 'hpf_18', 'stage_10', or just 10)
            key = stage

            # Extract generated cells for this stage - [T, B, D] -> take slice at position
            generated_cells = generated_seq[context_len + i, :].cpu()  # [B, D]
            results[key] = generated_cells

        return results

    def generate_scheduling_matrix(
        self, horizon: int, scheduling_type: str, uncertainty_scale: float
    ) -> torch.Tensor:
        """
        Generate scheduling matrix for different denoising strategies.

        Args:
            horizon: Number of target positions to generate
            scheduling_type: Type of schedule ("pyramid", "full_sequence", "autoregressive", "trapezoid")
            uncertainty_scale: Scale factor for temporal uncertainty

        Returns:
            Scheduling matrix with shape (num_steps, horizon)
        """
        sampling_timesteps = self.model.sampling_timesteps

        match scheduling_type:
            case "pyramid":
                return self._generate_pyramid_scheduling_matrix(
                    horizon, uncertainty_scale, sampling_timesteps
                )
            case "full_sequence":
                # All positions denoise together
                return np.arange(sampling_timesteps, -1, -1)[:, None].repeat(
                    horizon, axis=1
                )
            case "autoregressive":
                # Strict sequential with maximum uncertainty
                return self._generate_pyramid_scheduling_matrix(
                    horizon, sampling_timesteps, sampling_timesteps
                )
            case "trapezoid":
                return self._generate_trapezoid_scheduling_matrix(
                    horizon, uncertainty_scale, sampling_timesteps
                )
            case _:
                raise ValueError(f"Unknown scheduling type: {scheduling_type}")

    def _generate_pyramid_scheduling_matrix(
        self, horizon: int, uncertainty_scale: float, sampling_timesteps: int
    ):
        """
        Generate pyramid scheduling matrix for target positions only.

        Args:
            horizon: Number of target positions
            uncertainty_scale: Scale for temporal uncertainty
            sampling_timesteps: Number of sampling timesteps

        Returns:
            Pyramid scheduling matrix
        """
        height = sampling_timesteps + int((horizon - 1) * uncertainty_scale)
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)

        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = (
                    sampling_timesteps + int(t * uncertainty_scale) - m
                )

        scheduling_matrix = np.clip(scheduling_matrix, 0, sampling_timesteps)

        # Schedule pruning: remove redundant rows where all positions have same noise level
        unique_rows = [0]
        for i in range(1, height):
            if not np.array_equal(scheduling_matrix[i], scheduling_matrix[i - 1]):
                unique_rows.append(i)

        return scheduling_matrix[unique_rows]

    def _generate_trapezoid_scheduling_matrix(
        self, horizon: int, uncertainty_scale: float, sampling_timesteps: int
    ):
        """
        Generate trapezoid scheduling matrix.

        Args:
            horizon: Number of target positions
            uncertainty_scale: Scale for temporal uncertainty
            sampling_timesteps: Number of sampling timesteps

        Returns:
            Trapezoid scheduling matrix
        """
        height = sampling_timesteps + int((horizon + 1) // 2 * uncertainty_scale)
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)

        for m in range(height):
            for t in range((horizon + 1) // 2):
                scheduling_matrix[m, t] = (
                    sampling_timesteps + int(t * uncertainty_scale) - m
                )
                scheduling_matrix[m, -t] = (
                    sampling_timesteps + int(t * uncertainty_scale) - m
                )

        return np.clip(scheduling_matrix, 0, sampling_timesteps)

    def _decode_all_latents(
        self,
        latents: np.ndarray,
        library_sizes: torch.Tensor,
        batch_size: int,
        vae_model: object,
        vae_type: str = "scvi",
        atac_library_sizes: torch.Tensor = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Decode latent representations to gene expression (and accessibility) space in batches.

        Args:
            latents: Latent representations [n_cells, latent_dim]
            library_sizes: RNA library sizes for decoding [n_cells]
            batch_size: Batch size for decoding
            vae_model: VAE model (scVI or MultiVI)
            vae_type: Type of VAE model ('scvi' or 'multivi')
            atac_library_sizes: ATAC library sizes for decoding [n_cells] (MultiVI only)

        Returns:
            (rna_matrix, atac_matrix) where:
                - rna_matrix: Gene expression matrix [n_cells, n_genes]
                - atac_matrix: Accessibility matrix [n_cells, n_regions] (None for scVI)
        """
        n_cells = len(latents)
        all_genes = []
        all_atac = [] if vae_type == "multivi" else None

        # Process in batches for memory efficiency
        with torch.no_grad():
            for i in range(0, n_cells, batch_size):
                end = min(i + batch_size, n_cells)
                batch_latents = latents[i:end]
                batch_lib_sizes = library_sizes[i:end]

                # Convert to tensors and move to device
                device = next(vae_model.module.parameters()).device
                z = torch.tensor(batch_latents, dtype=torch.float32).to(device)
                batch_index = torch.zeros(len(batch_latents), dtype=torch.int64).to(device)
                batch_lib_sizes = batch_lib_sizes.to(device)

                # Call generative with model-specific parameters
                if vae_type == "multivi":
                    # DiF outputs are samples, not posterior means. qz_m=None makes this explicit.
                    # use_z_mean must be False (assertion guards against accidental changes).
                    use_z_mean = False
                    assert use_z_mean is False, "use_z_mean must be False when qz_m=None"

                    outputs = vae_model.module.generative(
                        z=z,
                        qz_m=None,
                        batch_index=batch_index,
                        cont_covs=None,
                        cat_covs=None,
                        libsize_expr=batch_lib_sizes,
                        use_z_mean=use_z_mean,
                        label=None,
                        transform_batch=None,
                    )
                    # MultiVI: Sample from ZINB distribution for RNA
                    from scvi.distributions import ZeroInflatedNegativeBinomial

                    # CRITICAL: Broadcast px_r to match batch dimension
                    px_r = outputs["px_r"]
                    if px_r.dim() == 1:
                        px_r = px_r.unsqueeze(0).expand_as(outputs["px_rate"])

                    px_dist = ZeroInflatedNegativeBinomial(
                        mu=outputs["px_rate"],
                        theta=px_r,
                        zi_logits=outputs["px_dropout"],
                    )
                    batch_genes = px_dist.sample().cpu().numpy()

                    # === ATAC Decoding (MultiVI only) ===
                    # Formula: Bernoulli(p * libsize_acc * region_factors)
                    # Matches MultiVI training objective (see scvi.module._multivae:886)
                    p = outputs["p"]  # Base accessibility probability [batch, n_regions]
                    batch_atac_lib = atac_library_sizes[i:end].to(device)

                    # Apply region-specific factors (learned during MultiVI training)
                    reg_factor = torch.sigmoid(vae_model.module.region_factors)

                    # Compute full Bernoulli parameter
                    atac_prob = p * batch_atac_lib * reg_factor
                    atac_prob = torch.clamp(atac_prob, 0, 1)

                    # Sample from Bernoulli to get binary accessibility
                    batch_atac = torch.bernoulli(atac_prob).cpu().numpy()
                    all_atac.append(batch_atac)
                else:
                    # scVI generative
                    outputs = vae_model.module.generative(
                        z=z,
                        batch_index=batch_index,
                        library=batch_lib_sizes,  # Note: library for scVI
                        cat_covs=None,
                    )
                    # scVI returns px distribution, sample from it
                    batch_genes = outputs["px"].sample().cpu().numpy()

                all_genes.append(batch_genes)

        rna_matrix = np.vstack(all_genes)
        atac_matrix = np.vstack(all_atac) if all_atac is not None else None
        return rna_matrix, atac_matrix

    def _prepare_context_cells(
        self, context_stages, train_stages, num_samples, verbose, data_manager
    ):
        """
        Prepare real cells as context for data_start mode.

        Args:
            context_stages: list of stages to use as context
            train_stages: list of training stages
            num_samples: Number of samples per stage
            verbose: Whether to print progress

        Returns:
            Tensor of shape [context_len, num_samples, latent_dim]
        """
        context_cells_list = []

        for stage in context_stages:
            # Use test data for evaluation context
            # (In training validation, this would use validation data)
            adata = data_manager.test_adata

            stage_col = data_manager.data_cfg.columns.stage_real
            mask = adata.obs[stage_col] == stage
            stage_cells = torch.tensor(
                adata.obsm["X_latent"][mask], dtype=torch.float32, device=self.device
            )

            # Sample num_samples cells for this stage
            if len(stage_cells) >= num_samples:
                idx = torch.randperm(len(stage_cells))[:num_samples]
            else:
                # If not enough cells, sample with replacement
                idx = torch.randint(0, len(stage_cells), (num_samples,))

            context_cells_list.append(stage_cells[idx])

        # Stack to create structured context: [context_len, num_samples, latent_dim]
        context_cells = torch.stack(context_cells_list, dim=0)  # dim=0 for [T, B, D]

        if verbose:
            log_print(
                f"\nUsing real data as initial context: shape {context_cells.shape}"
            )

        return context_cells

    def generate_and_save_samples(
        self,
        gen_config: GenerationConfig,
        model_config: ModelConfig,
        gen_data: GenerationData,
    ) -> Path:
        """
        Generate samples and save them as AnnData files.

        Args:
            gen_config: Generation configuration (mode, num_samples, etc.)
            model_config: Model configuration (uncertainty_scale, chunk_size)
            gen_data: Data required for generation (data_manager, vae_model, output_dir)

        Returns:
            Path to samples directory containing generated.h5ad and real.h5ad
        """
        # Extract from configs for cleaner code
        mode = gen_config.mode
        target_stages = gen_config.target_stages
        num_samples = gen_config.num_samples
        batch_size = gen_config.batch_size
        decode_batch_size = gen_config.decode_batch_size
        verbose = gen_config.verbose

        uncertainty_scale = model_config.uncertainty_scale
        chunk_size = model_config.chunk_size

        data_manager = gen_data.data_manager
        vae_model = gen_data.vae_model
        output_dir = gen_data.output_dir

        # Check prerequisites
        if data_manager is None:
            raise ValueError("DataManager required for generation.")
        if vae_model is None:
            raise ValueError("VAE model required for generation.")
        if output_dir is None:
            raise ValueError("Output directory required for generation.")

        # Detect VAE type early to use throughout
        vae_type = "multivi" if "MULTIVI" in type(vae_model).__name__ else "scvi"

        # Get all unique stages
        stage_col = data_manager.data_cfg.columns.stage_real
        train_stages = sorted(data_manager.train_adata.obs[stage_col].unique())
        test_stages = sorted(data_manager.test_adata.obs[stage_col].unique())

        if target_stages is not None:
            all_stages = target_stages
        else:
            all_stages = sorted(set(train_stages + test_stages))

        # Use output directory directly (already created by caller)
        inference_dir = output_dir

        if verbose:
            log_print(f"\nGenerating samples for {len(all_stages)} stages...")
            log_print(f"Mode: {mode}")
            log_print(f"Samples per stage: {num_samples}")
            if num_samples > batch_size:
                log_print(
                    f"Batch size: {batch_size} (will generate in {(num_samples + batch_size - 1) // batch_size} batches)"
                )

        # Prepare generation based on mode
        if mode == "noise_start":
            # Generate all stages from noise - no context needed
            # chunk_size is flexible and handled in generate_future_cells
            context_cells = None
            context_stages = None
            stages_to_generate = all_stages
        elif mode == "data_start":
            # Use real cells as context, flexible chunk_size during inference
            # chunk_size determines how many stages to generate
            # Context is everything except the last chunk_size stages

            if len(all_stages) < 2:
                raise ValueError(
                    f"Need at least 2 stages for data_start mode. Only {len(all_stages)} provided."
                )

            # Simple logic: generate the last chunk_size stages, use the rest as context
            num_context = len(all_stages) - chunk_size

            # Ensure we have at least 1 context stage
            if num_context < 1:
                num_context = 1
                if verbose:
                    log_print(
                        f"Warning: Not enough stages for chunk_size={chunk_size}. Using {num_context} context stage(s)."
                    )

            context_stages = all_stages[:num_context]
            stages_to_generate = all_stages[num_context:]

            context_cells = self._prepare_context_cells(
                context_stages, train_stages, num_samples, verbose, data_manager
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Generate samples in batches to avoid OOM
        generated_all = {}
        num_batches = (num_samples + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, num_samples)
            batch_samples = batch_end - batch_start

            if verbose and num_batches > 1:
                log_print(
                    f"  Generating batch {batch_idx + 1}/{num_batches} ({batch_samples} samples)..."
                )

            # Prepare context for this batch if needed
            batch_context = None
            if context_cells is not None:
                batch_context = context_cells[:, batch_start:batch_end, :]

            # Generate this batch
            batch_generated = self.generate_future_cells(
                context_cells=batch_context,
                context_stages=context_stages,
                target_stages=stages_to_generate,
                num_samples=batch_samples,
                uncertainty_scale=uncertainty_scale,
                stabilize_context=True,  # Explicit instead of relying on default
                context_noise_scale=model_config.context_noise_scale,  # From config
                verbose=verbose and batch_idx == 0,  # Only verbose for first batch
                chunk_size=chunk_size,
                scheduling_type=model_config.scheduling_matrix,  # From config
            )

            # Merge batch results
            for stage, data in batch_generated.items():
                if stage not in generated_all:
                    generated_all[stage] = []
                generated_all[stage].append(
                    data.cpu()
                )  # Move to CPU to save GPU memory

            # Clean up memory between batches
            if batch_idx < num_batches - 1:  # Don't clean after last batch
                del batch_generated
                torch.cuda.empty_cache()

        # Concatenate all batches
        for stage in generated_all:
            generated_all[stage] = torch.cat(generated_all[stage], dim=0)

        # Collect all generated cells and metadata
        all_generated_latents = []
        all_generated_stages = []
        all_real_latents = []
        all_real_genes = []  # Store actual gene expression, not reconstructed
        all_real_atac = []  # Store actual ATAC accessibility (MultiVI only)
        all_real_stages = []  # Still needed for latents file
        all_real_obs = []  # Store all obs metadata for real cells

        for stage in stages_to_generate:
            # Generated data
            gen_latents = generated_all[stage].cpu().numpy()
            all_generated_latents.append(gen_latents)
            all_generated_stages.extend([stage] * gen_latents.shape[0])

            # Real data - use the correct dataset based on which stage it is
            # Check if this stage is in train or test
            if stage in set(train_stages):
                adata_to_use = data_manager.train_adata  # Use train data
            else:
                adata_to_use = data_manager.test_adata  # Use test data

            real_mask = adata_to_use.obs[stage_col] == stage
            if real_mask.sum() > 0:
                # Get all real cells for this stage (no sampling here)
                real_indices = np.where(real_mask)[0]

                # Get real latents for real_latents.h5ad
                real_latents = adata_to_use.obsm["X_latent"][real_indices]
                all_real_latents.append(real_latents)

                # Get actual raw gene expression for real_gex.h5ad
                # Use raw_counts layer if available, otherwise use X
                if "raw_counts" in adata_to_use.layers:
                    real_genes = adata_to_use.layers["raw_counts"][real_indices]
                    # Convert sparse to dense if needed
                    if hasattr(real_genes, "toarray"):
                        real_genes = real_genes.toarray()
                else:
                    real_genes = adata_to_use.X[real_indices]
                    if hasattr(real_genes, "toarray"):
                        real_genes = real_genes.toarray()

                all_real_genes.append(real_genes)
                all_real_stages.extend([stage] * len(real_indices))
                all_real_obs.append(adata_to_use.obs.iloc[real_indices])

        # Collect real ATAC data for MultiVI (must be done after loop to know vae_type)
        if vae_type == "multivi":
            for stage in stages_to_generate:
                if stage in set(train_stages):
                    mdata_to_use = data_manager.train_mdata
                else:
                    mdata_to_use = data_manager.test_mdata

                real_mask = mdata_to_use.mod['rna'].obs[stage_col] == stage
                real_indices = np.where(real_mask)[0]

                if len(real_indices) > 0:
                    atac_data = mdata_to_use.mod['atac']
                    real_atac = atac_data.X[real_indices]
                    if hasattr(real_atac, "toarray"):
                        real_atac = real_atac.toarray()

                    # Binarize: counts > 0 → 1, else 0 (matches MultiVI training target)
                    real_atac = (real_atac > 0).astype(np.float32)
                    all_real_atac.append(real_atac)

        # Create AnnData objects
        if verbose:
            log_print("\nDecoding latents to gene expression...")

        # Stack latents
        generated_latents = np.vstack(all_generated_latents)
        real_latents = np.vstack(all_real_latents)

        # Sample library sizes for decoding - use per-stage pools
        def sample_lib_sizes_per_stage(stages: np.ndarray) -> torch.Tensor:
            if len(stages) == 0:
                return torch.tensor([]).reshape(0, 1)

            all_libs = []
            for stage in stages:
                stage_lib_pool = self.stage_to_lib_sizes[stage]
                idx = torch.randint(0, len(stage_lib_pool), (1,))
                all_libs.append(stage_lib_pool[idx])

            return torch.cat(all_libs, dim=0).reshape(-1, 1)

        gen_lib_sizes = sample_lib_sizes_per_stage(all_generated_stages)

        # Sample ATAC library sizes for MultiVI
        if vae_type == "multivi":
            def sample_atac_lib_sizes_per_stage(stages: np.ndarray) -> torch.Tensor:
                if len(stages) == 0:
                    return torch.tensor([]).reshape(0, 1)

                all_libs = []
                for stage in stages:
                    stage_lib_pool = self.stage_to_atac_lib_sizes[stage]
                    idx = torch.randint(0, len(stage_lib_pool), (1,))
                    all_libs.append(stage_lib_pool[idx])

                return torch.cat(all_libs, dim=0).reshape(-1, 1)

            gen_atac_lib_sizes = sample_atac_lib_sizes_per_stage(all_generated_stages)
        else:
            gen_atac_lib_sizes = None

        # Decode to gene expression (and accessibility for MultiVI)
        generated_genes, generated_atac = self._decode_all_latents(
            generated_latents,
            gen_lib_sizes,
            batch_size=decode_batch_size,
            vae_model=vae_model,
            vae_type=vae_type,
            atac_library_sizes=gen_atac_lib_sizes,
        )
        # Use the actual raw gene expression we collected, not decoded
        real_genes = np.vstack(all_real_genes)

        if verbose:
            log_print(f"Decoded to gene expression: {generated_genes.shape[1]} genes")
            if vae_type == "multivi":
                log_print(f"Decoded to accessibility: {generated_atac.shape[1]} regions")

        # Get gene metadata from training data
        # Both train and test have the same genes and metadata
        gene_metadata = data_manager.train_adata.var.copy()

        # Save 4 separate files
        # 1. Generated latents
        generated_latent_adata = ad.AnnData(X=generated_latents)
        stage_col_gen = data_manager.data_cfg.columns.stage_generated
        generated_latent_adata.obs[stage_col_gen] = all_generated_stages
        generated_latent_adata.obs[stage_col_gen] = generated_latent_adata.obs[
            stage_col_gen
        ].astype("category")
        generated_latent_adata.write(inference_dir / "generated_latents.h5ad")

        # 2. Generated gene expression
        generated_gex_adata = ad.AnnData(X=generated_genes)
        generated_gex_adata.obs[stage_col_gen] = all_generated_stages
        generated_gex_adata.obs[stage_col_gen] = generated_gex_adata.obs[
            stage_col_gen
        ].astype("category")
        generated_gex_adata.var = gene_metadata.copy()  # This also sets var_names
        generated_gex_adata.write(inference_dir / "generated_gex.h5ad")

        # 3. Real latents
        real_latent_adata = ad.AnnData(X=real_latents)
        real_latent_adata.obs[stage_col] = all_real_stages
        real_latent_adata.obs[stage_col] = real_latent_adata.obs[stage_col].astype(
            "category"
        )
        real_latent_adata.write(inference_dir / "real_latents.h5ad")

        # 4. Real gene expression
        real_gex_adata = ad.AnnData(X=real_genes)
        real_gex_adata.obs = pd.concat(all_real_obs, ignore_index=True)
        real_gex_adata.var = gene_metadata.copy()  # This also sets var_names
        real_gex_adata.write(inference_dir / "real_gex.h5ad")

        # 5. Generated ATAC (MultiVI only)
        if vae_type == "multivi":
            generated_atac_adata = ad.AnnData(X=generated_atac)
            generated_atac_adata.obs[stage_col_gen] = all_generated_stages
            generated_atac_adata.obs[stage_col_gen] = generated_atac_adata.obs[
                stage_col_gen
            ].astype("category")
            atac_metadata = data_manager.train_mdata.mod['atac'].var.copy()
            generated_atac_adata.var = atac_metadata.copy()
            generated_atac_adata.write(inference_dir / "generated_atac.h5ad")

        # 6. Real ATAC (MultiVI only)
        if vae_type == "multivi":
            real_atac_stacked = np.vstack(all_real_atac)
            real_atac_adata = ad.AnnData(X=real_atac_stacked)
            real_atac_adata.obs = pd.concat(all_real_obs, ignore_index=True)
            atac_metadata = data_manager.train_mdata.mod['atac'].var.copy()
            real_atac_adata.var = atac_metadata.copy()
            real_atac_adata.write(inference_dir / "real_atac.h5ad")

        if verbose:
            log_print(
                f"\nSaved {generated_latent_adata.shape[0]} generated cells to {inference_dir}/generated_latents.h5ad"
            )
            log_print(
                f"Saved {real_latent_adata.shape[0]} real cells to {inference_dir}/real_latents.h5ad"
            )
            if vae_type == "multivi":
                log_print(
                    f"Saved {generated_atac_adata.shape[0]} generated ATAC cells to {inference_dir}/generated_atac.h5ad"
                )
                log_print(
                    f"Saved {real_atac_adata.shape[0]} real ATAC cells to {inference_dir}/real_atac.h5ad"
                )

        return inference_dir
