"""Shared utilities for VAE models (scVI and MultiVI)."""

import hashlib
from pathlib import Path
from typing import Any

import torch

from cellpace.utils.common import _get_stage_range, save_vae_metadata_to_pickle
from cellpace.utils.logging import setup_logger

logger = setup_logger("vae_base")


class VAEUtils:
    """Shared utilities for scVI and MultiVI models."""

    @staticmethod
    def get_model_hash(model_path: Path) -> str:
        """
        Compute SHA-1 hash of model.pt file for reproducibility tracking.

        Works for both scVI and MultiVI models.

        Args:
            model_path: Path to model directory

        Returns:
            First 8 characters of the model.pt SHA-1 hash
        """
        model_pt = model_path / "model.pt"
        if not model_pt.exists():
            raise FileNotFoundError(f"model.pt not found at {model_pt}")

        sha1 = hashlib.sha1()
        with open(model_pt, "rb") as f:
            while chunk := f.read(65536):
                sha1.update(chunk)
        return sha1.hexdigest()[:8]

    @staticmethod
    def save_metadata_with_per_stage_libs(model: Any, data_manager: Any, output_dir: Path, config: Any) -> None:
        """
        Save VAE metadata with per-stage encoder-inferred library sizes.

        Auto-detects scVI vs MultiVI based on model type.
        Includes both training and test stages (using frozen encoder on test data).

        Args:
            model: Trained scVI or MultiVI model
            data_manager: DataManager instance
            output_dir: Directory to save metadata.pkl
            config: Configuration object

        Returns:
            dict: Saved metadata
        """
        logger.info("\nSaving metadata with per-stage encoder-inferred library sizes...")

        stage_col = config.data.columns.stage_real
        train_stages = sorted(data_manager.train_adata.obs[stage_col].unique())
        test_stages = (
            sorted(data_manager.test_adata.obs[stage_col].unique())
            if len(data_manager.test_adata) > 0
            else []
        )

        is_multimodal = "MULTIVI" in type(model).__name__

        total_stages = len(train_stages) + len(test_stages)
        logger.info(f"  [green]✓[/green] Computing library sizes for {total_stages} stages")

        if is_multimodal:
            stage_to_rna_lib = {}
            stage_to_atac_lib = {}

            for stage in train_stages:
                stage_mask = data_manager.train_mdata.mod['rna'].obs[stage_col] == stage
                stage_mdata = data_manager.train_mdata[stage_mask]
                rna_lib, atac_lib = VAEUtils._compute_multivi_encoder_libs(model, stage_mdata)
                stage_to_rna_lib[stage] = rna_lib
                stage_to_atac_lib[stage] = atac_lib

            for stage in test_stages:
                stage_mask = data_manager.test_mdata.mod['rna'].obs[stage_col] == stage
                stage_mdata = data_manager.test_mdata[stage_mask]
                rna_lib, atac_lib = VAEUtils._compute_multivi_encoder_libs(model, stage_mdata)
                stage_to_rna_lib[stage] = rna_lib
                stage_to_atac_lib[stage] = atac_lib

            library_sizes_dict = {
                "stage_to_rna_lib_sizes": stage_to_rna_lib,
                "stage_to_atac_lib_sizes": stage_to_atac_lib,
            }

        else:
            stage_to_lib = {}

            for stage in train_stages:
                stage_mask = data_manager.train_adata.obs[stage_col] == stage
                stage_adata = data_manager.train_adata[stage_mask]
                stage_lib = VAEUtils._compute_scvi_encoder_libs(model, stage_adata)
                stage_to_lib[stage] = stage_lib

            for stage in test_stages:
                stage_mask = data_manager.test_adata.obs[stage_col] == stage
                stage_adata = data_manager.test_adata[stage_mask]
                stage_lib = VAEUtils._compute_scvi_encoder_libs(model, stage_adata)
                stage_to_lib[stage] = stage_lib

            library_sizes_dict = {"stage_to_lib_sizes": stage_to_lib}

        stage_min, stage_max = _get_stage_range(train_stages)

        metadata = save_vae_metadata_to_pickle(
            output_dir,
            train_stages,
            test_stages,
            library_sizes_dict,
            stage_min,
            stage_max,
        )

        logger.info(f"  Train stages: {len(train_stages)}, Test stages: {len(test_stages)}")
        logger.info(f"  Saved per-stage library sizes for {len(train_stages) + len(test_stages)} total stages")

        return metadata

    @staticmethod
    def _compute_scvi_encoder_libs(model, adata) -> torch.Tensor:
        """
        Compute encoder-inferred library sizes for scVI (RNA only).

        Args:
            model: Trained scvi.model.SCVI instance
            adata: AnnData with gene expression

        Returns:
            torch.Tensor: Encoder-inferred library sizes
        """
        device = next(model.module.parameters()).device
        all_libsize = []
        batch_size = 512

        for i in range(0, len(adata), batch_size):
            batch_adata = adata[i : i + batch_size]

            x = torch.tensor(
                (
                    batch_adata.X.toarray()
                    if hasattr(batch_adata.X, "toarray")
                    else batch_adata.X
                ),
                dtype=torch.float32,
            ).to(device)
            batch_index = torch.zeros(len(batch_adata), dtype=torch.int64, device=device)

            with torch.no_grad():
                inference_outputs = model.module.inference(x, batch_index, n_samples=1)
                all_libsize.append(inference_outputs["library"].cpu())

        return torch.cat(all_libsize, dim=0).squeeze()

    @staticmethod
    def _compute_multivi_encoder_libs(model, mdata) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute encoder-inferred library sizes for MultiVI (RNA + ATAC).

        Args:
            model: Trained scvi.model.MULTIVI instance
            mdata: MuData with RNA and ATAC modalities

        Returns:
            (rna_libsize, atac_libsize): tuple of torch.Tensors
        """
        device = next(model.module.parameters()).device
        all_libsize_expr = []
        all_libsize_acc = []
        batch_size = 512

        for i in range(0, len(mdata), batch_size):
            batch_mdata = mdata[i:i+batch_size]

            rna_X = torch.tensor(
                batch_mdata.mod['rna'].X.toarray() if hasattr(batch_mdata.mod['rna'].X, 'toarray')
                else batch_mdata.mod['rna'].X, dtype=torch.float32
            ).to(device)
            atac_X = torch.tensor(
                batch_mdata.mod['atac'].X.toarray() if hasattr(batch_mdata.mod['atac'].X, 'toarray')
                else batch_mdata.mod['atac'].X, dtype=torch.float32
            ).to(device)

            x_combined = torch.cat([rna_X, atac_X], dim=1)
            y = torch.zeros(len(batch_mdata), 1, device=device)
            batch_index = torch.zeros(len(batch_mdata), dtype=torch.int64, device=device)
            cell_idx = torch.arange(len(batch_mdata), device=device)

            with torch.no_grad():
                inference_outputs = model.module.inference(
                    x_combined, y, batch_index,
                    cont_covs=None, cat_covs=None,
                    label=torch.zeros(len(batch_mdata), 1, dtype=torch.long, device=device),
                    cell_idx=cell_idx,
                    size_factor=None
                )

                all_libsize_expr.append(inference_outputs['libsize_expr'].cpu())
                all_libsize_acc.append(inference_outputs['libsize_acc'].cpu())

        rna_libsize = torch.cat(all_libsize_expr, dim=0).squeeze()
        atac_libsize = torch.cat(all_libsize_acc, dim=0).squeeze()

        return rna_libsize, atac_libsize
