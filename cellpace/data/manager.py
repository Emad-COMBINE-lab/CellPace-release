"""Data manager coordinating preprocessing, VAE setup, and DiF dataset creation."""

# Standard library
from pathlib import Path
from typing import Any

# Third-party
import anndata as ad
from mudata import MuData

# This prevents FutureWarnings about automatic obs/var pulling
import mudata as md
import numpy as np
import scvi
from omegaconf import OmegaConf

md.set_options(pull_on_update=False)

# Local imports (after mudata config)
from ..utils.common import _get_stage_range
from ..utils.logging import setup_logger
from .dif import DiFPregenDataset
from .vae import DataPreprocessor, MuDataProcessor
from .validator import DataValidator

logger = setup_logger("data_manager")


class DataManager:
    """
    Simplified data manager that coordinates between DataPreprocessor and model training.

    Responsibilities:
    - Initialize DataPreprocessor or MuDataProcessor based on input format
    - Combine train/val when validation split exists
    - Setup data for scVI or MultiVI
    - Create DiF datasets
    - Save metadata for inference
    """

    def __init__(self, data_cfg, seed: int = 42, vae_type: str = "scvi") -> None:
        """Initialize with config views and load data.

        Args:
            data_cfg: Data configuration from config file
            seed: Random seed for reproducibility
            vae_type: Type of VAE to use ('scvi' or 'multivi')
        """
        self.data_cfg = data_cfg
        self.seed = seed
        self.vae_type = vae_type
        self._initialize_processor()
        self._setup_data()

    def _initialize_processor(self) -> None:
        """Initialize DataPreprocessor or MuDataProcessor based on input format."""
        data_path = Path(self.data_cfg.paths.input)

        # Detect if input is MuData (.h5mu) or AnnData (.h5ad)
        is_mudata = data_path.suffix == ".h5mu"

        if is_mudata:
            logger.info("Detected MuData input - using MuDataProcessor for multiome data")
            self._initialize_mudata_processor()
            return

        # Otherwise use standard AnnData processor
        logger.info("Detected AnnData input - using DataPreprocessor")

        strategy = self.data_cfg.split.strategy
        DataValidator.validate_strategy(strategy, ["stage_holdout"])

        if strategy == "stage_holdout":
            # OmegaConf will raise clear error if stage_holdout doesn't exist
            # Use to_container to recursively convert ListConfig to list
            stage_splits = OmegaConf.to_container(
                self.data_cfg.split.stage_holdout, resolve=True
            )


        special_genes = None
        if hasattr(self.data_cfg, "special_genes_to_include"):
            special_genes = (
                list(self.data_cfg.special_genes_to_include)
                if self.data_cfg.special_genes_to_include
                else None
            )


        layer_key = self.data_cfg.vae_use_layer


        self.processor = DataPreprocessor(
            data_path=str(data_path),
            output_dir=str(Path(self.data_cfg.paths.preprocess_cache)),
            target_genes=self.data_cfg.n_genes,
            stage_splits=stage_splits,
            special_genes=special_genes,
            time_key=self.data_cfg.columns.stage_real,
            layer_key=layer_key,
            seed=self.seed,
        )

    def _initialize_mudata_processor(self) -> None:
        """Initialize MuDataProcessor for multiome data."""
        data_path = Path(self.data_cfg.paths.input)


        strategy = self.data_cfg.split.strategy
        DataValidator.validate_strategy(strategy, ["stage_holdout"])

        if strategy == "stage_holdout":
            stage_splits = OmegaConf.to_container(
                self.data_cfg.split.stage_holdout, resolve=True
            )


        self.processor = MuDataProcessor(
            mudata_path=str(data_path),
            output_dir=str(Path(self.data_cfg.paths.preprocess_cache)),
            time_key=self.data_cfg.columns.stage_real,
            stage_splits=stage_splits,
            seed=self.seed,
        )

    def _setup_data(self) -> None:
        """Setup train/test data from processor."""

        # Helper to concat adatas while preserving var metadata
        def concat_preserve_var(adatas: list) -> Any:
            var_metadata = adatas[0].var.copy()
            combined = ad.concat(adatas, axis=0)
            combined.var = var_metadata
            return combined

        # Helper to concat MuData while preserving structure
        def concat_mudata(mdatas: list) -> Any:
            # Concatenate each modality separately
            rna_list = [m.mod['rna'] for m in mdatas]
            atac_list = [m.mod['atac'] for m in mdatas]

            combined_rna = concat_preserve_var(rna_list)
            combined_atac = concat_preserve_var(atac_list)


            # Note: Global option pull_on_update=False set at module import
            # This means we don't automatically pull obs/var from modalities
            # For MultiVI, each modality's obs/var is sufficient
            combined_mdata = MuData({'rna': combined_rna, 'atac': combined_atac})

            return combined_mdata


        result = self.processor.get_train_test_splits()
        if self.processor.has_val_split:
            train_adata, val_adata, test_adata = result
            self.train_adata = concat_preserve_var([train_adata, val_adata])
            self.test_adata = test_adata
        else:
            self.train_adata, self.test_adata = result

        logger.info(f"\n[bold]Training {self.vae_type.upper()}[/bold]")
        logger.info(f"  [green]✓[/green] Setup: {self.train_adata.n_obs:,} cells (train+val)")


        self.train_adata.obs["data_split"] = "train"
        self.test_adata.obs["data_split"] = "test"
        self.all_adata = concat_preserve_var([self.train_adata, self.test_adata])


        stage_col = self.data_cfg.columns.stage_real
        self.all_stages = sorted(self.all_adata.obs[stage_col].unique())

        # Handle MuData if using MuDataProcessor
        if isinstance(self.processor, MuDataProcessor):

            mdata_result = self.processor.get_mudata_splits()
            if self.processor.has_val_split:
                train_mdata, val_mdata, test_mdata = mdata_result
                self.train_mdata = concat_mudata([train_mdata, val_mdata])
                self.test_mdata = test_mdata
            else:
                self.train_mdata, self.test_mdata = mdata_result

        # Validate consistency: MultiVI requires multiome data
        DataValidator.validate_vae_type_consistency(
            vae_type=self.vae_type,
            has_multiome_data=isinstance(self.processor, MuDataProcessor),
            data_path=Path(self.data_cfg.paths.input)
        )

        # Setup based on VAE type (single source of truth)
        if self.vae_type == "multivi":
            logger.info("Setting up MuData for MultiVI training...")
            self.setup_for_multivi(self.train_mdata)
            logger.info(
                f"  [green]✓[/green] MultiVI setup: {self.train_mdata.n_obs} cells (RNA+ATAC)"
            )
        else:
            self.setup_for_scvi(self.train_adata)

    def get_all(self) -> tuple:
        """Get train, test, combined data and stage lists."""
        return self.train_adata, self.test_adata, self.all_adata, self.all_stages

    def setup_for_scvi(self, adata: Any) -> None:
        """
        Setup an AnnData object for scVI if not already setup.

        This method is idempotent - safe to call multiple times on the same adata.
        Can be used for:
        - Training scVI models
        - Loading pre-trained scVI models
        - Preparing any new AnnData for scVI operations

        Args:
            adata: AnnData object to setup for scVI

        Returns:
            adata: The same AnnData object (for chaining)
        """
        # Ensure X contains raw counts (scVI expectation)
        if "raw_counts" in adata.layers:
            X_data = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
            layer_data = (
                adata.layers["raw_counts"].toarray()
                if hasattr(adata.layers["raw_counts"], "toarray")
                else adata.layers["raw_counts"]
            )

            if not np.array_equal(X_data, layer_data):
                logger.info("Copying raw counts to X matrix for scVI")
                adata.X = adata.layers["raw_counts"].copy()

        if "_scvi_uuid" not in adata.uns:
            logger.debug(f"Setting up AnnData (n_obs={adata.n_obs}) for scVI")
            scvi.model.SCVI.setup_anndata(
                adata,
                layer="raw_counts",
                labels_key=None,
                categorical_covariate_keys=None,
                continuous_covariate_keys=None,
            )
        return adata

    def setup_for_multivi(self, mdata: Any) -> None:
        """
        Setup a MuData object for MultiVI if not already setup.

        This method is idempotent - safe to call multiple times on the same mdata.
        Can be used for:
        - Training MultiVI models
        - Loading pre-trained MultiVI models
        - Preparing any new MuData for MultiVI operations

        Args:
            mdata: MuData object to setup for MultiVI

        Returns:
            mdata: The same MuData object (for chaining)
        """

        if "_scvi_uuid" not in mdata.uns:
            logger.debug(f"Setting up MuData (n_obs={mdata.n_obs}) for MultiVI")
            scvi.model.MULTIVI.setup_mudata(
                mdata,
                rna_layer=None,  # Use .X (should contain raw counts)
                atac_layer=None,  # Use .X (should contain raw counts)
                batch_key=None,  # No batch correction
                categorical_covariate_keys=None,  # Don't regress out covariates
                continuous_covariate_keys=None,
                modalities={
                    "rna_layer": "rna",
                    "atac_layer": "atac",
                },
            )
        return mdata

    def create_dif_dataset(self, latent_adata: Any, vae_hash: str | None = None) -> Any:
        """
        Create pregenerated DiF dataset from latent scVI adata.

        Args:
            latent_adata: AnnData with latents, expects them in obsm['X_latent']
            vae_hash: SHA-1 hash of scVI model to track changes

        Returns:
            DiFPregenDataset: Pregenerated dataset with cached sequences
        """
        if "X_latent" not in latent_adata.obsm:
            logger.info("Moving latents from X to obsm['X_latent'] for DiF dataset")
            latent_adata.obsm["X_latent"] = latent_adata.X


        stage_col = self.data_cfg.columns.stage_real
        train_stages = sorted(self.train_adata.obs[stage_col].unique())
        stage_min, stage_max = _get_stage_range(train_stages)

        logger.info(
            f"Creating DiF dataset with {self.data_cfg.sequences.n_sequences} sequences..."
        )

        dataset = DiFPregenDataset(
            adata=latent_adata,
            latents=latent_adata.obsm["X_latent"],
            seq_len=self.data_cfg.sequences.length,
            n_sequences=self.data_cfg.sequences.n_sequences,
            min_cells_threshold=self.data_cfg.sequences.min_cells_per_stage,
            stage_col=stage_col,
            stage_min=stage_min,
            stage_max=stage_max,
            cache_dir=Path(self.data_cfg.paths.dif_seq_cache),
            regenerate=self.data_cfg.sequences.force_regenerate,
            seed=self.seed,
            vae_hash=vae_hash,
        )

        return dataset
