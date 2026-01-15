#!/usr/bin/env python3
"""Subsample demo data to create a lightweight example dataset."""

from pathlib import Path
import anndata as ad
import numpy as np
import scanpy as sc

script_dir = Path(__file__).parent
cells_per_stage = 50
n_top_genes = 100


def select_hvg(adata, n_genes):
    """Select top n highly variable genes."""
    adata_copy = adata.copy()
    sc.pp.normalize_total(adata_copy, target_sum=1e4)
    sc.pp.log1p(adata_copy)
    sc.pp.highly_variable_genes(adata_copy, n_top_genes=n_genes)
    return adata[:, adata_copy.var["highly_variable"]].copy()


def subsample_by_stage(adata, stage_key, n_per_stage):
    """Subsample AnnData by taking n cells per stage."""
    indices = []
    for stage in adata.obs[stage_key].unique():
        stage_mask = adata.obs[stage_key] == stage
        stage_indices = np.where(stage_mask)[0]
        n_take = min(n_per_stage, len(stage_indices))
        indices.extend(stage_indices[:n_take])
        print(f"  {stage}: {len(stage_indices)} -> {n_take}")
    return adata[sorted(indices)].copy()


# ============================================================================
# Subsample retinal progenitor
# ============================================================================
print("Processing prcd_retinal_progenitor...")
adata = ad.read_h5ad(script_dir / "prcd_retinal_progenitor.h5ad")
print(f"  Original: {adata.shape[0]} cells x {adata.shape[1]} genes")

stage_col = "author_somite_count"

# Select top HVGs
print(f"  Selecting top {n_top_genes} HVGs...")
adata = select_hvg(adata, n_top_genes)
print(f"  After HVG: {adata.shape[0]} cells x {adata.shape[1]} genes")

# Subsample cells per stage
adata_sub = subsample_by_stage(adata, stage_col, cells_per_stage)

adata_sub.write_h5ad(script_dir / "demo_retinal_progenitor.h5ad")
print(f"  Final: {adata_sub.shape[0]} cells x {adata_sub.shape[1]} genes")
