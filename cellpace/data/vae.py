"""Data preprocessing with HVG selection and train/val/test splitting."""

# Standard library
import pickle
import time
from pathlib import Path
from typing import Any

# Third-party
from mudata import read_h5mu
import numpy as np
import scanpy as sc

# Local imports
from ..utils.common import format_path_for_display, set_random_seeds
from ..utils.logging import log_config_panel, setup_logger
from .validator import DataValidator

logger = setup_logger("data_processor")


class DataPreprocessor:
    """Complete data handler for temporal scRNA-seq - no model dependencies."""

    def __init__(
        self,
        data_path: str,
        output_dir: str | None = None,
        time_key: str = "author_somite_count",
        layer_key: str = "raw_counts",
        target_genes: int = 3000,
        force_reprocess: bool = False,
        seed: int = 42,
        stage_splits: dict[str, list[str | None]] = None,
        special_genes: list[str | None] = None,
    ):
        """
        Initialize the DataPreprocessor.

        Args:
            data_path: Path to the h5ad data file
            output_dir: Directory to save processed data (optional)
            time_key: Column name for stage/timepoint labels
            layer_key: Layer key for raw counts
            target_genes: Number of highly variable genes to select
            force_reprocess: Whether to force reprocessing even if cached data exists
            seed: Random seed for reproducibility (default: 42)
            stage_splits: dict with 'train', 'val_interp', 'val_extrap', 'test_interp', 'test_extrap' stage lists
            special_genes: list of gene names to always include (e.g., perturbation targets)
        """
        # Set random seed for reproducibility
        set_random_seeds(seed)
        self.seed = seed
        # Also set scanpy's random seed
        sc.settings.seed = seed

        # Validate stage splits provided
        DataValidator.validate_stage_splits(stage_splits)

        # Determine if we have validation split from config (not from loaded data)
        val_stages = stage_splits.get("val_interp", []) + stage_splits.get(
            "val_extrap", []
        )
        self.has_val_split = len(val_stages) > 0

        # Store parameters
        self.data_path = Path(data_path)
        self.output_dir = (
            Path(output_dir) if output_dir else self.data_path.parent / "cache"
        )
        self.time_key = time_key
        self.layer_key = layer_key
        self.target_genes = target_genes
        self.stage_splits = stage_splits
        self.special_genes = special_genes or []

        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use output_dir directly as cache directory
        self.cache_dir = self.output_dir

        log_config_panel(
            "DataPreprocessor Configuration",
            {
                "Data": format_path_for_display(self.data_path),
                "Output": format_path_for_display(self.output_dir),
                "Strategy": "stage_holdout",
                "Genes": f"{self.target_genes:,} HVGs",
                "Seed": str(self.seed),
            },
        )

        # Set up cache file paths
        self.train_cache_path = self.cache_dir / "train_data.h5ad"
        self.test_cache_path = self.cache_dir / "test_data.h5ad"
        self.val_cache_path = (
            self.cache_dir / "val_data.h5ad"
        )  # Optional validation split
        self.metadata_cache_path = self.cache_dir / "metadata.pkl"

        # Initialize data containers
        self.train_adata = None
        self.val_adata = None  # Optional validation split
        self.test_adata = None
        self.train_indices = None
        self.val_indices = None  # Optional validation indices
        self.test_indices = None

        # Check cache and process if needed
        if self._check_cache() and not force_reprocess:
            logger.info("Found valid cached data - loading from cache...")
            self._load_cache()
        else:
            if force_reprocess:
                logger.info("Force reprocess enabled - processing data from scratch...")
            else:
                logger.info("No valid cache found - processing data from scratch...")
            self._process_data_with_cache()

    def _check_cache(self) -> bool:
        """Check if valid cached data exists."""
        # Build list of required cache files
        required_files = [
            self.train_cache_path,
            self.test_cache_path,
            self.metadata_cache_path,
        ]
        if self.has_val_split:
            required_files.append(self.val_cache_path)

        # Check if all required files exist
        if not DataValidator.validate_cache_exists(required_files):
            return False

        # Check if cache is older than raw data (only if raw data exists)
        if self.data_path.exists():
            raw_data_mtime = self.data_path.stat().st_mtime
            cache_mtime = min(
                self.train_cache_path.stat().st_mtime,
                self.test_cache_path.stat().st_mtime,
                self.metadata_cache_path.stat().st_mtime,
            )

            if raw_data_mtime > cache_mtime:
                logger.info("Cache is older than raw data, will reprocess")
                return False

        # Load metadata and check parameters
        expected_params = {
            "target_genes": self.target_genes,
            "time_key": self.time_key,
        }

        if not DataValidator.validate_cache_metadata(
            self.metadata_cache_path, expected_params
        ):
            logger.info("Cache parameters don't match, will reprocess")
            return False

        logger.info("Found valid cache")
        return True

    def _process_data_with_cache(self):
        """Process raw data with stage-manual splitting and HVG selection."""
        logger.info("\n[bold]Data Processing[/bold]")

        # Step 1 & 2: Load and validate data
        adata = sc.read_h5ad(str(self.data_path))
        DataValidator.validate_adata_fields(adata, [self.layer_key], [self.time_key])
        logger.info(
            f"  [green]✓[/green] Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes"
        )

        # Step 3: Stage-manual splitting - extract stage lists
        def get_stages(prefix: str) -> list[str]:
            return self.stage_splits.get(
                f"{prefix}_interp", []
            ) + self.stage_splits.get(f"{prefix}_extrap", [])

        train_s = self.stage_splits["train"]
        val_s = get_stages("val")
        test_s = get_stages("test")

        # Compute indices for each split
        self.train_indices = np.where(adata.obs[self.time_key].isin(train_s))[0]
        self.val_indices = np.where(adata.obs[self.time_key].isin(val_s))[0]
        self.test_indices = np.where(adata.obs[self.time_key].isin(test_s))[0]

        logger.info(
            f"  [green]✓[/green] Split: {len(self.train_indices):,} train / {len(self.val_indices):,} val / {len(self.test_indices):,} test "
            f"({len(train_s) + len(val_s) + len(test_s)} total stages)"
        )

        # Step 4: Prepare data structure
        adata.layers["log_norm"] = adata.X.copy()  # Save log-normalized
        adata.X = adata.layers[self.layer_key].copy()  # Use raw counts

        # Step 5: HVG selection on training data with special genes
        adata_train = adata[self.train_indices].copy()
        adata_train.X = adata_train.layers["log_norm"].copy()  # Use log-norm for HVG
        sc.pp.highly_variable_genes(
            adata_train, n_top_genes=self.target_genes, flavor="seurat", subset=False
        )
        adata_train.X = adata_train.layers["raw_counts"].copy()  # Restore raw
        hvg_genes = adata_train.var_names[adata_train.var.highly_variable].tolist()

        # Add special genes if provided
        hvg_genes = self._add_special_genes(hvg_genes, adata_train.var_names.tolist())
        logger.info(f"  [green]✓[/green] Selected {len(hvg_genes):,} HVGs")

        # Filter to selected genes
        adata = adata[:, hvg_genes].copy()

        adata.uns["train_indices"] = self.train_indices
        adata.uns["test_indices"] = self.test_indices
        if self.has_val_split:
            adata.uns["val_indices"] = self.val_indices
        adata.var["highly_variable"] = True

        # Step 6: Create final splits
        train_adata = adata[self.train_indices].copy()
        test_adata = adata[self.test_indices].copy()
        val_adata = adata[self.val_indices].copy() if self.has_val_split else None

        # Helper to add task labels (interp/extrap)
        def add_task_labels(split_adata: Any, extrap_stages: list[str]) -> None:
            split_adata.obs["task"] = "interp"  # Default
            for stage in extrap_stages:
                split_adata.obs.loc[split_adata.obs[self.time_key] == stage, "task"] = (
                    "extrap"
                )

        # Add task labels for val and test
        if self.has_val_split:
            add_task_labels(val_adata, self.stage_splits.get("val_extrap", []))
            add_task_labels(test_adata, self.stage_splits.get("test_extrap", []))

        # Store in class attributes
        self.train_adata = train_adata
        self.val_adata = val_adata
        self.test_adata = test_adata

        # Save to cache
        self._save_cache(train_adata, test_adata, val_adata)
        logger.info(
            f"  [green]✓[/green] Cached to {format_path_for_display(self.cache_dir)}"
        )

    def _add_special_genes(
        self, hvg_genes: list[str], all_genes: list[str]
    ) -> list[str]:
        """Add special genes to HVG list if they exist in the dataset."""
        if not self.special_genes:
            return hvg_genes

        genes_to_add = [
            g for g in self.special_genes if g in all_genes and g not in hvg_genes
        ]
        if genes_to_add:
            logger.info(
                f"  Adding {len(genes_to_add)} special genes: {genes_to_add[:5]}{'...' if len(genes_to_add) > 5 else ''}"
            )
        return hvg_genes + genes_to_add

    def _load_cache(self) -> None:
        """Load cached processed data."""
        logger.info("\n[bold]Loading Cached Data[/bold]")

        # Helper function to load H5AD with retries for concurrent access
        def load_h5ad_with_retry(
            path: str, max_retries: int = 10, delay: int = 5
        ) -> Any:
            for attempt in range(max_retries):
                try:
                    return sc.read_h5ad(path)
                except BlockingIOError as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"File locked, waiting {delay}s before retry {attempt+1}/{max_retries}: {path}"
                        )
                        time.sleep(delay)
                    else:
                        raise e

        train_adata = load_h5ad_with_retry(self.train_cache_path)
        test_adata = load_h5ad_with_retry(self.test_cache_path)
        val_adata = (
            load_h5ad_with_retry(self.val_cache_path) if self.has_val_split else None
        )

        with open(self.metadata_cache_path, "rb") as f:
            metadata = pickle.load(f)

        self.train_indices = metadata.get("train_indices")
        self.test_indices = metadata.get("test_indices")
        if self.has_val_split:
            self.val_indices = metadata.get("val_indices")

        self.train_adata = train_adata
        self.val_adata = val_adata
        self.test_adata = test_adata

        val_info = f", {val_adata.n_obs:,} val" if self.has_val_split else ""
        logger.info(
            f"  [green]✓[/green] Loaded {train_adata.n_obs:,} train{val_info}, {test_adata.n_obs:,} test | "
            f"{train_adata.n_vars:,} genes"
        )

    def _save_cache(
        self,
        train_adata: sc.AnnData,
        test_adata: sc.AnnData,
        val_adata: sc.AnnData | None = None,
    ) -> None:
        """Save processed data to cache."""
        # Save data files
        train_adata.write_h5ad(self.train_cache_path)
        test_adata.write_h5ad(self.test_cache_path)

        if val_adata is not None:
            val_adata.write_h5ad(self.val_cache_path)

        # Build metadata
        metadata = {
            "target_genes": self.target_genes,
            "time_key": self.time_key,
            "train_indices": self.train_indices,
            "test_indices": self.test_indices,
        }

        # Add validation indices if present
        if self.has_val_split:
            metadata["val_indices"] = self.val_indices

        with open(self.metadata_cache_path, "wb") as f:
            pickle.dump(metadata, f)

    def get_train_test_splits(
        self,
    ) -> tuple[sc.AnnData, sc.AnnData] | tuple[sc.AnnData, sc.AnnData, sc.AnnData]:
        """Get train/test/val split AnnData objects for scVI and evaluation.

        Returns:
            - With validation split: (train_adata, val_adata, test_adata)
            - Without validation split: (train_adata, test_adata)
        """
        if self.has_val_split:
            return self.train_adata, self.val_adata, self.test_adata
        else:
            return self.train_adata, self.test_adata

    def get_split_statistics(self) -> dict:
        """Get statistics about train/test/val splits.

        Returns:
            Dictionary with split statistics (n_genes, n_train_cells, etc.)
        """
        stats = {
            "n_genes": self.train_adata.n_vars,
            "n_train_cells": self.train_adata.n_obs,
            "n_test_cells": self.test_adata.n_obs,
            "time_key": self.time_key,
        }

        if self.has_val_split:
            stats["n_val_cells"] = (
                self.val_adata.n_obs if self.val_adata is not None else 0
            )

        return stats


# ============================================================================
# MuData Processor for Multiome Data
# ============================================================================
class MuDataProcessor:
    """
    Data handler for multiome (RNA + ATAC) data in MuData format.

    Handles loading MuData and creating train/val/test splits for MultiVI training.
    Stores separate MuData objects for each split (train, val, test) to enable
    proper data handling and prevent data leakage.

    Design: Similar to DataPreprocessor but for MuData instead of AnnData.
    """

    def __init__(
        self,
        mudata_path: str,
        output_dir: str | None = None,
        time_key: str = "palantir_discrete",
        seed: int = 42,
        stage_splits: dict[str, list[str | None]] = None,
    ):
        """
        Initialize MuDataProcessor.

        Args:
            mudata_path: Path to .h5mu file
            output_dir: Directory for cache
            time_key: Column name for pseudotime bins
            seed: Random seed
            stage_splits: dict with train/val/test stage lists
        """
        set_random_seeds(seed)
        self.seed = seed

        # Validate stage splits
        DataValidator.validate_stage_splits(stage_splits)

        # Determine if we have validation split
        val_stages = stage_splits.get("val_interp", []) + stage_splits.get(
            "val_extrap", []
        )
        self.has_val_split = len(val_stages) > 0

        # Store parameters
        self.mudata_path = Path(mudata_path)
        self.output_dir = (
            Path(output_dir) if output_dir else self.mudata_path.parent / "cache"
        )
        self.time_key = time_key
        self.stage_splits = stage_splits

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.output_dir

        # Define cache paths
        self.train_cache_path = self.cache_dir / "train_mdata.h5mu"
        self.test_cache_path = self.cache_dir / "test_mdata.h5mu"
        self.val_cache_path = self.cache_dir / "val_mdata.h5mu"
        self.metadata_cache_path = self.cache_dir / "metadata.pkl"

        logger.info("\n=== MuDataProcessor Configuration ===")
        logger.info(f"MuData: {self.mudata_path}")
        logger.info(f"Time key: {self.time_key}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 42)

        # Load and split data (with caching)
        if self._load_cache():
            logger.info("  [green]✓[/green] Loaded from cache")
        else:
            logger.info("Processing from raw data...")
            self._load_and_split()
            self._save_cache()

    def _load_and_split(self):
        """Load MuData and create train/val/test splits."""
        logger.info("Loading MuData...")
        mdata = read_h5mu(self.mudata_path)
        logger.info(f"  Loaded: {mdata.n_obs} cells")
        logger.info(f"  Modalities: {list(mdata.mod.keys())}")
        logger.info(f"    RNA:  {mdata.mod['rna'].shape}")
        logger.info(f"    ATAC: {mdata.mod['atac'].shape}")

        # Validate time labels exist and are consistent across modalities
        logger.info(f"Validating time labels ('{self.time_key}')...")
        DataValidator.validate_mudata_time_labels(
            mdata, time_key=self.time_key, modalities=["rna", "atac"]
        )
        logger.info("  [green]✓[/green] Time labels validated: RNA and ATAC consistent")

        # Extract splits based on time_key
        def get_stages(prefix: str) -> list[str]:
            return self.stage_splits.get(
                f"{prefix}_interp", []
            ) + self.stage_splits.get(f"{prefix}_extrap", [])

        train_s = self.stage_splits["train"]
        val_s = get_stages("val")
        test_s = get_stages("test")

        # Get indices (time_key is in modality obs, not mdata.obs)
        time_col = mdata.mod["rna"].obs[self.time_key]
        train_idx = np.where(time_col.isin(train_s))[0]
        val_idx = np.where(time_col.isin(val_s))[0]
        test_idx = np.where(time_col.isin(test_s))[0]

        logger.info(
            f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"
        )

        # Store MuData for each split separately (prevents data leakage)
        self.train_mdata = mdata[train_idx].copy()
        self.val_mdata = mdata[val_idx].copy() if self.has_val_split else None
        self.test_mdata = mdata[test_idx].copy()

        # Also store RNA-only AnnData for DiF training
        # This allows DiF to work with RNA latents from MultiVI
        self.train_adata = mdata.mod["rna"][train_idx].copy()
        self.val_adata = (
            mdata.mod["rna"][val_idx].copy() if self.has_val_split else None
        )
        self.test_adata = mdata.mod["rna"][test_idx].copy()

        # Add task labels to RNA splits
        def add_task_labels(adata: Any, extrap_stages: list[str]) -> None:
            adata.obs["task"] = "interp"
            for stage in extrap_stages:
                adata.obs.loc[adata.obs[self.time_key] == stage, "task"] = "extrap"

        if self.has_val_split:
            add_task_labels(self.val_adata, self.stage_splits.get("val_extrap", []))
        add_task_labels(self.test_adata, self.stage_splits.get("test_extrap", []))

        logger.info("  [green]✓[/green] MuData loaded and split")

    def get_train_test_splits(self) -> tuple:
        """
        Get RNA-only train/val/test splits for DiF training.

        Returns:
            tuple of (train_adata, val_adata, test_adata) if has_val_split,
            or (train_adata, test_adata) otherwise
        """
        if self.has_val_split:
            return self.train_adata, self.val_adata, self.test_adata
        else:
            return self.train_adata, self.test_adata

    def get_mudata_splits(self) -> tuple:
        """
        Get MuData train/val/test splits for MultiVI training.

        Returns:
            tuple of (train_mdata, val_mdata, test_mdata) if has_val_split,
            or (train_mdata, test_mdata) otherwise
        """
        if self.has_val_split:
            return self.train_mdata, self.val_mdata, self.test_mdata
        else:
            return self.train_mdata, self.test_mdata

    def _save_cache(self):
        """Save processed MuData splits to cache."""
        logger.info(f"Saving processed MuData to cache: {self.cache_dir}")

        # Save MuData files
        self.train_mdata.write_h5mu(self.train_cache_path)
        self.test_mdata.write_h5mu(self.test_cache_path)

        if self.has_val_split and self.val_mdata is not None:
            self.val_mdata.write_h5mu(self.val_cache_path)

        # Build metadata
        metadata = {
            "time_key": self.time_key,
            "stage_splits": self.stage_splits,
            "has_val_split": self.has_val_split,
            "seed": self.seed,
        }

        with open(self.metadata_cache_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"  [green]✓[/green] Saved MuData cache to {self.cache_dir}")

    def _load_cache(self):
        """
        Load processed MuData splits from cache if valid.

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        # Check if all required files exist
        required_files = [
            self.train_cache_path,
            self.test_cache_path,
            self.metadata_cache_path,
        ]

        if self.has_val_split:
            required_files.append(self.val_cache_path)

        if not DataValidator.validate_cache_exists(required_files):
            return False

        # Check if cache is older than raw data
        if self.mudata_path.exists():
            raw_mtime = self.mudata_path.stat().st_mtime
            cache_mtime = min(
                self.train_cache_path.stat().st_mtime,
                self.test_cache_path.stat().st_mtime,
                self.metadata_cache_path.stat().st_mtime,
            )

            if raw_mtime > cache_mtime:
                logger.info("Cache is older than raw data, will reprocess")
                return False

        # Load and validate metadata
        with open(self.metadata_cache_path, "rb") as f:
            metadata = pickle.load(f)

        # Validate cache matches current config
        if metadata.get("time_key") != self.time_key:
            logger.info(
                f"Cache time_key mismatch: {metadata.get('time_key')} vs {self.time_key}"
            )
            return False

        if metadata.get("stage_splits") != self.stage_splits:
            logger.info("Cache stage_splits mismatch, will reprocess")
            return False

        # Load MuData splits from cache
        logger.info("Loading MuData splits from cache...")
        self.train_mdata = read_h5mu(self.train_cache_path)
        self.test_mdata = read_h5mu(self.test_cache_path)

        if self.has_val_split:
            self.val_mdata = read_h5mu(self.val_cache_path)
        else:
            self.val_mdata = None

        # Also load RNA-only AnnData splits (stored in MuData modalities)
        self.train_adata = self.train_mdata.mod["rna"].copy()
        self.test_adata = self.test_mdata.mod["rna"].copy()

        if self.has_val_split and self.val_mdata is not None:
            self.val_adata = self.val_mdata.mod["rna"].copy()
        else:
            self.val_adata = None

        logger.info(f"  Train: {self.train_mdata.n_obs} cells")
        logger.info(f"  Test: {self.test_mdata.n_obs} cells")
        if self.has_val_split:
            logger.info(f"  Val: {self.val_mdata.n_obs} cells")

        return True
