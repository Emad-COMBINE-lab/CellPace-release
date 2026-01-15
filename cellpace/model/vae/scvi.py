"""scVI model wrapper for encoding cells to latent space."""

import warnings
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import scvi
import torch
from anndata import AnnData
from omegaconf import DictConfig

from cellpace.model.vae.base import VAEUtils
from cellpace.utils.logging import setup_logger

logger = setup_logger("train_scvi")

matplotlib.use("agg")
warnings.filterwarnings("ignore", message=".*CUDA device.*Tensor Cores.*")
warnings.filterwarnings("ignore", message=".*LOCAL_RANK.*CUDA_VISIBLE_DEVICES.*")


class SCVIModel:
    """scVI model implementation."""

    def __init__(self, output_dir) -> None:
        """Initialize scVI model."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model: Any | None = None  # Will be SCVI model once loaded
        self.model_name = "scVI"

    def save(self) -> bool:
        """Save the trained model."""
        if self.model is None:
            logger.info("No model to save!")
            return False

        save_path = self.output_dir
        logger.info(f"Saving {self.model_name} model to {save_path}")
        self.model.save(save_path, overwrite=True)

        return True

    def load(self, adata: AnnData | None = None) -> bool:
        """
        Load a trained model.

        Args:
            adata: AnnData object to use for loading (required for most models)

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        load_path = self.output_dir
        model_file = load_path / "model.pt"

        if not model_file.exists():
            logger.info(f"No model found at {load_path}")
            return False

        logger.info(f"Loading {self.model_name} model from {load_path}")

        # If adata is provided, try loading with it
        if adata is not None:
            # DataManager has already set up the AnnData, use it directly
            logger.info("Loading model with provided AnnData...")

            # Get the model class from subclass implementation
            model_class = self._get_model_class()

            def load_model_with_adata() -> bool:
                self.model = model_class.load(str(load_path), adata=adata)

            success = self._load_model(load_model_with_adata)
            return success

        logger.info("Failed to load model - no AnnData provided")
        return False

    def _load_model(self, load_func: Any) -> bool:
        """
        Load a PyTorch model.

        Args:
            load_func: Function to call for loading the model

        Returns:
            bool: True if successful
        """
        load_func()
        logger.info("Model loaded successfully!")
        return True

    def get_latent(self, adata: AnnData) -> np.ndarray:
        """
        Get latent representation.

        Args:
            adata: AnnData object to get latent representation for

        Returns:
            np.ndarray: Latent representation
        """
        if self.model is None:
            logger.info("Model not trained/loaded!")
            return None

        # Use AnnData directly - DataManager has already set it up
        # No need to copy since we're just reading
        return self.model.get_latent_representation(adata)

    def _get_model_class(self) -> Any:
        """
        Get the model class for this model type.
        To be implemented by subclasses.
        """
        return scvi.model.SCVI

    def generate_from_latents(self, latents: np.ndarray, library_sizes: np.ndarray) -> np.ndarray:
        """
        Generate gene expression from provided latent vectors.

        This is a cleaner interface that doesn't require dummy adata.

        Args:
            latents: np.ndarray of shape (n_cells, latent_dim)
            library_sizes: Required torch.Tensor of shape (n_cells, 1).
                          Must be provided from metadata.

        Returns:
            np.ndarray: Generated gene expression of shape (n_cells, n_genes)
        """
        if self.model is None:
            logger.info("Model not trained/loaded!")
            return None

        if library_sizes is None:
            raise ValueError(
                "library_sizes must be provided. Get them from metadata.pkl "
                "(global_train_lib_sizes or global_test_lib_sizes)."
            )

        n_cells = latents.shape[0]

        # Convert latents to tensor if needed
        if not isinstance(latents, torch.Tensor):
            latents = torch.tensor(latents).float()

        # Generate expression using the model's generative function
        with torch.no_grad():
            batch_index = torch.zeros(n_cells, dtype=torch.int64)
            generated = self.model.module.generative(
                latents,
                batch_index=batch_index,
                library=library_sizes,
                cat_covs=None,
            )
            expression = generated["px"].sample().cpu().numpy()

        return expression


def train_scvi(config: DictConfig, data_manager: Any) -> str:
    """
    Standalone function to train scVI model - minimal modification from class method.

    Args:
        config: Configuration object from main.py (OmegaConf)
        data_manager: DataManager instance with loaded data

    Returns:
        str: Path to saved model
    """
    # Get config views
    experiment_cfg = config.experiment
    scvi_cfg = config.scvi

    # Setup output directory
    output_dir = Path(experiment_cfg.vae_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a SCVIModel instance to use its methods
    scvi_model_instance = SCVIModel(output_dir)

    # Check if model already exists
    if not scvi_cfg.training.retrain and scvi_model_instance.load(
        data_manager.train_adata
    ):
        logger.info(
            "  [green]✓[/green] Loaded existing trained model. Set scvi.training.retrain=true to train from scratch."
        )
        return str(output_dir)

    logger.info("Training scVI model...")

    # DataManager has already set up the AnnData for scVI, so use it directly
    # No need for _prepare_adata since:
    # - DataManager ensures raw counts are in X
    # - DataManager calls setup_anndata
    # - We use metadata.pkl for library sizes now

    # Log library size statistics for reference
    lib_sizes = data_manager.train_adata.X.sum(axis=1)
    lib_sizes = lib_sizes.A1 if hasattr(lib_sizes, "A1") else lib_sizes
    logger.info(
        f"Library size stats: mean={lib_sizes.mean():.1f}, "
        f"std={lib_sizes.std():.1f}, "
        f"min={lib_sizes.min():.1f}, "
        f"max={lib_sizes.max():.1f}"
    )

    # Initialize the scVI model (from the scvi library) using config parameters
    arch_cfg = scvi_cfg.architecture
    scvi_vae_model = scvi.model.SCVI(
        adata=data_manager.train_adata,
        n_hidden=arch_cfg.n_hidden,
        n_latent=arch_cfg.n_latent,
        n_layers=arch_cfg.n_layers,
        dropout_rate=arch_cfg.dropout_rate,
        gene_likelihood=arch_cfg.gene_likelihood,
    )
    # Store in the instance for saving later
    scvi_model_instance.model = scvi_vae_model

    logger.info("Model initialized!")
    logger.info(f"Model parameters: {scvi_vae_model}")

    # Always use GPU for training
    logger.info("Using CUDA GPU for training")

    # Train the scVI VAE model using its built-in train method
    train_cfg = scvi_cfg.training
    scvi_vae_model.train(
        max_epochs=train_cfg.max_epochs,
        batch_size=train_cfg.batch_size,
        accelerator="cuda",
        devices=1,
        early_stopping=train_cfg.early_stopping.enabled,
        plan_kwargs={"weight_decay": train_cfg.weight_decay},
    )

    logger.info("scVI training complete!")

    # Save the model
    scvi_model_instance.save()

    # Save metadata with per-stage encoder-inferred library sizes
    VAEUtils.save_metadata_with_per_stage_libs(
        scvi_model_instance.model, data_manager, output_dir, config
    )

    return str(output_dir)
