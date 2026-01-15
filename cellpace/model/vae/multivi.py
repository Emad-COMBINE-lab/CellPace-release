"""MultiVI model wrapper for multiome (RNA + ATAC) data."""

from functools import wraps
from pathlib import Path
from typing import Any

import scvi
import torch
from omegaconf import DictConfig

from cellpace.model.vae.base import VAEUtils
from cellpace.utils.logging import setup_logger

logger = setup_logger("train_multivi")


def _patch_multivi_train() -> None:
    """
    Monkey patch scvi.model.MULTIVI.train() to accept early_stopping_patience.

    Issue: scvi-tools hardcodes early_stopping_patience=50 in MultiVI.train():
        runner = self._train_runner_cls(..., early_stopping_patience=50, **kwargs)

    Unlike n_epochs_kl_warmup (a parameter of train()), early_stopping_patience
    must go through **kwargs, but conflicts with the hardcoded value.

    Solution: Extract it from kwargs before calling original train(),
    then inject it by temporarily wrapping _train_runner_cls.
    """
    original_train = scvi.model.MULTIVI.train

    @wraps(original_train)
    def patched_train(self, *args, **kwargs) -> Any:
        early_stopping_patience = kwargs.pop("early_stopping_patience", 50)
        original_runner_cls = self._train_runner_cls

        def patched_runner_cls(*runner_args, **runner_kwargs) -> Any:
            runner_kwargs["early_stopping_patience"] = early_stopping_patience
            return original_runner_cls(*runner_args, **runner_kwargs)

        self._train_runner_cls = patched_runner_cls
        try:
            return original_train(self, *args, **kwargs)
        finally:
            self._train_runner_cls = original_runner_cls

    scvi.model.MULTIVI.train = patched_train


def train_multivi(config: DictConfig, data_manager: Any) -> Path:
    """
    Train MultiVI model on MuData with paired RNA + ATAC.

    Args:
        config: OmegaConf configuration
        data_manager: DataManager instance with train_mdata prepared

    Returns:
        Path to saved model directory
    """
    # Apply monkey patch to enable configurable early_stopping_patience
    _patch_multivi_train()

    multivi_cfg = config.multivi
    exp_cfg = config.experiment
    output_dir = Path(exp_cfg.vae_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training data: {data_manager.train_mdata.n_obs} cells")
    logger.info(f"  RNA:  {data_manager.train_mdata.mod['rna'].shape}")
    logger.info(f"  ATAC: {data_manager.train_mdata.mod['atac'].shape}")

    # Check if model already exists and we don't want to retrain
    if not multivi_cfg.training.retrain and output_dir.exists():
        try:
            logger.info(f"Checking for existing MultiVI model at {output_dir}")
            model = scvi.model.MULTIVI.load(
                str(output_dir), adata=data_manager.train_mdata
            )
            model_hash = VAEUtils.get_model_hash(output_dir)
            logger.info(f" Loaded existing MultiVI model (hash: {model_hash})")
            return output_dir
        except FileNotFoundError:
            logger.info("Model directory exists but model files not found. Training new model...")
        except ValueError as e:
            logger.info(f"Model incompatible with current data: {e}. Training new model...")
        except (RuntimeError, OSError) as e:
            logger.warning(f"Error loading model ({type(e).__name__}): {e}. Training new model...")

    logger.info("\nInitializing MultiVI model...")
    arch_cfg = multivi_cfg.architecture

    model = scvi.model.MULTIVI(
        data_manager.train_mdata,
        n_hidden=arch_cfg.n_hidden,
        n_latent=arch_cfg.n_latent,
        n_layers_encoder=arch_cfg.n_layers,
        n_layers_decoder=arch_cfg.n_layers,
        dropout_rate=arch_cfg.dropout_rate,
        gene_likelihood=arch_cfg.gene_likelihood,
        modality_weights=arch_cfg.modality_weights,
        modality_penalty=arch_cfg.modality_penalty,
    )

    logger.info("MultiVI architecture:")
    logger.info(f"  Hidden units: {arch_cfg.n_hidden}")
    logger.info(f"  Latent dims: {arch_cfg.n_latent}")
    logger.info(f"  Encoder layers: {arch_cfg.n_layers}")
    logger.info(f"  Decoder layers: {arch_cfg.n_layers}")
    logger.info(f"  Dropout: {arch_cfg.dropout_rate}")
    logger.info(f"  Gene likelihood: {arch_cfg.gene_likelihood}")
    logger.info(f"  Modality weights: {arch_cfg.modality_weights}")
    logger.info(f"  Modality penalty: {arch_cfg.modality_penalty}")

    logger.info("\nStarting MultiVI training...")
    train_cfg = multivi_cfg.training
    logger.info(f"  Max epochs: {train_cfg.max_epochs}")
    logger.info(f"  Batch size: {train_cfg.batch_size}")
    logger.info(f"  Learning rate: {train_cfg.lr}")
    logger.info(f"  Early stopping: {train_cfg.early_stopping}")
    logger.info(f"  Early stopping patience: {train_cfg.early_stopping_patience}")
    logger.info(f"  KL warmup epochs: {train_cfg.n_epochs_kl_warmup}")

    train_kwargs = {
        "max_epochs": train_cfg.max_epochs,
        "batch_size": train_cfg.batch_size,
        "early_stopping": train_cfg.early_stopping,
        "n_epochs_kl_warmup": train_cfg.n_epochs_kl_warmup,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "lr": train_cfg.lr,
        "weight_decay": train_cfg.weight_decay,
    }

    if train_cfg.check_val_every_n_epoch is not None:
        train_kwargs["check_val_every_n_epoch"] = train_cfg.check_val_every_n_epoch

    # Handled by monkey patch to override scvi-tools hardcoded value
    train_kwargs["early_stopping_patience"] = train_cfg.early_stopping_patience

    model.train(**train_kwargs)

    logger.info(f"\nSaving MultiVI model to {output_dir}")
    model.save(str(output_dir), overwrite=True)

    # Save metadata with per-stage encoder-inferred library sizes
    VAEUtils.save_metadata_with_per_stage_libs(model, data_manager, output_dir, config)

    model_hash = VAEUtils.get_model_hash(output_dir)
    logger.info(f" MultiVI training complete! (hash: {model_hash})")
    logger.info("=" * 80)

    return output_dir
