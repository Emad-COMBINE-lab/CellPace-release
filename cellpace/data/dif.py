"""PyTorch Dataset classes for DiF temporal sequence training."""

# Standard library
from pathlib import Path

# Third-party
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset

# Local imports
from ..utils.common import _extract_stage_number
from ..utils.logging import setup_logger
from .validator import DataValidator

logger = setup_logger("dif_datasets")


class BaseDiFDataset:
    """
    Base class for DiF datasets with shared stage mapping and normalization logic.

    This class handles:
    - stage to cell index mapping
    - stage filtering by minimum cell count
    - stage normalization to [0, 1] range
    - Valid window computation for sequence generation
    """

    def __init__(
        self,
        adata,
        seq_len: int,
        min_cells_threshold: int,
        stage_col: str,
        stage_min: int,
        stage_max: int,
    ):
        self.seq_len = seq_len
        self.min_cells_threshold = min_cells_threshold
        self.stage_col = stage_col
        self.stage_min = stage_min
        self.stage_max = stage_max

        # Setup stage mapping and filtering
        self._setup_stage_mapping(adata)

    def _setup_stage_mapping(self, adata):
        """Setup stage to cell indices mapping."""

        # Group by stages
        stage_groups = adata.obs.groupby(self.stage_col, observed=False)
        all_stages = sorted(stage_groups.groups.keys())

        # Filter stages with enough cells
        valid_stages = []
        for stage in all_stages:
            cell_count = len(stage_groups.get_group(stage))
            if cell_count >= self.min_cells_threshold:
                valid_stages.append(stage)
            else:
                logger.info(
                    f"Warning: Filtering out {stage} - only {cell_count} cells (< {self.min_cells_threshold})"
                )

        self.all_stages = sorted(valid_stages, key=lambda x: _extract_stage_number(x))
        self.stage_to_indices = {
            s: np.where(adata.obs[self.stage_col] == s)[0] for s in valid_stages
        }
        logger.info(
            f"Valid stages after filtering: {len(valid_stages)}/{len(all_stages)}"
        )

        # Extract numeric values for normalization
        self.stage_to_num = {s: _extract_stage_number(s) for s in self.all_stages}

        # Validate normalization parameters
        DataValidator.validate_normalization_params(self.stage_min, self.stage_max)

        logger.info(
            f"Using training normalization range: [{self.stage_min}, {self.stage_max}]"
        )

        # Build list of valid window starting positions
        self.window_start_indices = list(range(len(self.all_stages) - self.seq_len + 1))

        # Compute geometric distribution parameter from observed gaps
        stage_nums = [self.stage_to_num[s] for s in self.all_stages]
        observed_gaps = np.diff(stage_nums)
        mean_gap = np.mean(observed_gaps)
        self.geometric_p = 1.0 / mean_gap
        logger.info(f"Geometric sampler: mean_gap={mean_gap:.2f}, p={self.geometric_p:.3f}")

    def normalize_stage_position(self, stage: str) -> float:
        """Normalize stage position to [0, 1] using training data range.

        Args:
            stage: Stage name (e.g., 'somite_10')

        Returns:
            Normalized position in [0, 1] range
        """
        num = self.stage_to_num[stage]
        return (num - self.stage_min) / (self.stage_max - self.stage_min)

    def compute_time_gaps(self, window_stages: list[str]) -> list[float]:
        """Compute time gaps between consecutive stages in a window."""
        gaps = [0.0]  # First position has no gap
        for i in range(1, len(window_stages)):
            gap = (
                self.stage_to_num[window_stages[i]]
                - self.stage_to_num[window_stages[i - 1]]
            )
            gaps.append(float(gap))
        return gaps

    def _sample_sequence_components(self):
        """
        Core logic for sampling a sequence window with varied gaps.
        Uses geometric distribution to naturally favor local coherence while allowing long jumps.
        """
        # This loop will continue until a valid sequence is generated
        while True:
            # The starting point must allow for the shortest possible sequence to complete.
            max_start_index = len(self.all_stages) - self.seq_len

            # Ensure max_start_index is not negative if the dataset is very small
            if max_start_index < 0:
                raise ValueError(
                    f"Cannot create a sequence of length {self.seq_len}. "
                    f"The number of valid stages ({len(self.all_stages)}) is too small."
                )

            start_index = np.random.randint(0, max_start_index + 1)

            window_indices = [start_index]
            current_index = start_index
            valid_sequence = True

            # Build the rest of the sequence with geometric distribution jumps
            for _ in range(self.seq_len - 1):
                max_jump_allowed = len(self.all_stages) - 1 - current_index

                # If no forward jump is possible, this sequence is invalid.
                if max_jump_allowed < 1:
                    valid_sequence = False
                    break

                # Sample from geometric distribution and clamp to valid range
                jump = max(1, min(np.random.geometric(p=self.geometric_p), max_jump_allowed))
                current_index += jump
                window_indices.append(current_index)

            # If the for loop completed without breaking, we have a valid sequence
            if valid_sequence:
                break  # Exit the while True loop

        # Once a valid sequence is found, proceed as before
        window_stages = [self.all_stages[i] for i in window_indices]
        cell_indices = [
            np.random.choice(self.stage_to_indices[s]) for s in window_stages
        ]

        return window_stages, cell_indices

    def _compute_sequence_features(
        self, window_stages: list[str]
    ) -> tuple[list[float], list[float]]:
        """
        Compute normalized positions and time gaps for a window.

        Returns:
            positions: Normalized stage positions
            gaps: Time gaps between consecutive stages
        """
        positions = [self.normalize_stage_position(s) for s in window_stages]
        gaps = self.compute_time_gaps(window_stages)
        return positions, gaps


class DiFPregenDataset(BaseDiFDataset, Dataset):
    """
    Pregenerated DiF dataset that caches sequences to disk for efficiency.

    Benefits:
    1. No redundant generation during training
    2. Consistent sequences across runs
    3. Efficient for hyperparameter tuning
    """

    def __init__(
        self,
        adata,
        latents: np.ndarray,
        seq_len: int,
        n_sequences: int,
        cache_dir: Path | None = None,
        regenerate: bool = False,
        seed: int = 42,
        min_cells_threshold: int = 100,
        stage_col: str = "author_somite_count",
        vae_hash: str | None = None,
        stage_min: int | None = None,
        stage_max: int | None = None,
    ):
        """
        Initialize pregenerated dataset.

        Args:
            adata: AnnData object with cell metadata
            latents: Latent representations [n_cells, latent_dim]
            seq_len: Length of sequences
            n_sequences: Number of sequences to pregenerate
            cache_dir: Directory to cache sequences
            regenerate: Force regeneration even if cache exists
            seed: Random seed
            min_cells_threshold: Minimum cells per stage
            stage_col: Column name for stage labels
            vae_hash: Hash of scVI model for cache validation
            stage_min: Minimum stage value for normalization (from training data)
            stage_max: Maximum stage value for normalization (from training data)
        """
        # Store dataset-specific attributes
        self.latents = latents
        self.n_sequences = n_sequences
        self.seed = seed
        self.vae_hash = vae_hash

        # Setup cache path
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_path = self.cache_dir / f"seq_{seq_len}_gap.pt"
        else:
            self.cache_path = None

        # Check if we should load from cache
        if not regenerate and self.cache_path and self.cache_path.exists():
            logger.info(f"Loading pregenerated sequences from {self.cache_path}")
            self._load_sequences(stage_min, stage_max)
        else:
            # Initialize base class
            super().__init__(
                adata, seq_len, min_cells_threshold, stage_col, stage_min, stage_max
            )

            # Generate new sequences
            logger.info(f"Generating {n_sequences} sequences...")
            self._generate_sequences()
            if self.cache_path:
                self._save_sequences()

    def _generate_sequences(self):
        """Generate all sequences at initialization."""
        np.random.seed(self.seed)

        sequences = []
        stage_positions = []
        stage_names = []
        time_gaps_list = []

        for i in tqdm.tqdm(range(self.n_sequences), desc="Generating sequences"):
            # Use base class method to get window and cell indices
            window_stages, cell_indices = self._sample_sequence_components()

            # Use base class method to compute features
            positions, gaps = self._compute_sequence_features(window_stages)

            # Get latent vectors (keep as numpy for efficiency)
            latent_seq = self.latents[np.array(cell_indices)]

            # Append to lists
            sequences.append(latent_seq)
            stage_positions.append(positions)
            time_gaps_list.append(gaps)
            stage_names.append(window_stages)

        # Convert to arrays
        self.sequences = np.array(sequences, dtype=np.float32)
        self.stage_positions = np.array(stage_positions, dtype=np.float32)
        self.time_gaps = np.array(time_gaps_list, dtype=np.float32)
        self.stage_names = stage_names

        logger.info(
            f"Generated {self.n_sequences} sequences of shape {self.sequences.shape}"
        )

    def _save_sequences(self):
        """Save pregenerated sequences to disk."""
        if self.cache_path is None:
            return

        data = {
            "sequences": torch.tensor(self.sequences),
            "stage_positions": torch.tensor(self.stage_positions),
            "time_gaps": torch.tensor(self.time_gaps),
            "stage_names": self.stage_names,
            "n_sequences": self.n_sequences,
            "seq_len": self.seq_len,
            "seed": self.seed,
            # Save normalization parameters to ensure consistency
            "stage_min": self.stage_min,
            "stage_max": self.stage_max,
            "vae_hash": self.vae_hash,
        }

        torch.save(data, self.cache_path)
        logger.info(f"Saved sequences to {self.cache_path}")

    def _load_sequences(self, expected_stage_min, expected_stage_max):
        """Load pregenerated sequences from disk and validate normalization."""
        data = torch.load(
            self.cache_path, weights_only=False
        )  # Need False for complex data structures

        # Check for required normalization parameters
        DataValidator.validate_cache_has_field(data, "stage_min", self.cache_path)
        DataValidator.validate_cache_has_field(data, "stage_max", self.cache_path)

        # Check scVI hash consistency
        DataValidator.validate_cache_has_field(data, "vae_hash", self.cache_path)
        DataValidator.validate_vae_hash_consistency(
            data["vae_hash"], self.vae_hash, self.cache_path
        )

        # Validate cached normalization matches expected parameters
        if expected_stage_min is not None and expected_stage_max is not None:
            DataValidator.validate_normalization_consistency(
                data["stage_min"],
                data["stage_max"],
                expected_stage_min,
                expected_stage_max,
                self.cache_path,
            )

        self.sequences = data["sequences"].numpy()
        self.stage_positions = data["stage_positions"].numpy()
        self.time_gaps = data["time_gaps"].numpy()
        self.stage_names = data["stage_names"]
        self.n_sequences = data["n_sequences"]
        self.stage_min = data["stage_min"]
        self.stage_max = data["stage_max"]

        logger.info(
            f"Loaded {self.n_sequences} sequences from cache with normalization [{self.stage_min}, {self.stage_max}]"
        )

    def __len__(self):
        """Return number of pregenerated sequences."""
        return self.n_sequences

    def __getitem__(self, idx):
        """Get pregenerated sequence by index."""
        return {
            "sequence": torch.tensor(self.sequences[idx], dtype=torch.float32),
            "stage_positions": torch.tensor(
                self.stage_positions[idx], dtype=torch.float32
            ),
            "time_gaps": torch.tensor(self.time_gaps[idx], dtype=torch.float32),
            "stages": self.stage_names[idx],
        }
