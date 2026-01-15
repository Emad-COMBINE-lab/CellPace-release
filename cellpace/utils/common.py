"""Common utilities for reproducibility and distributed training."""

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
import torch.distributed as dist

# Project root for relative path display
try:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
except (NameError, AttributeError):
    PROJECT_ROOT = Path.cwd()


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0) in distributed training."""
    if not dist.is_available():
        return True

    if not dist.is_initialized():
        rank = os.environ.get('LOCAL_RANK', None)
        if rank is not None:
            return int(rank) == 0
        rank = os.environ.get('RANK', None)
        if rank is not None:
            return int(rank) == 0
        return True

    return dist.get_rank() == 0


def format_path_for_display(path: str | Path) -> str:
    """
    Format file paths for user-friendly display.

    Converts absolute paths to relative paths from project root when possible.

    Args:
        path: Absolute or relative path

    Returns:
        Relative path string for display (e.g., 'demo/data/file.h5ad')
    """
    try:
        path = Path(path).resolve()
        rel_path = path.relative_to(PROJECT_ROOT)
        return str(rel_path)
    except (ValueError, RuntimeError):
        return str(path)


# Internal utility functions to reduce redundancy
def _extract_stage_number(stage_str: str) -> int:
    """Extract numeric value from stage/timepoint string (e.g., 'stage_10' -> 10, 'hpf_48' -> 48)."""
    if isinstance(stage_str, str) and "_" in stage_str:
        # Handle any 'prefix_XX' format (stage_10, hpf_48, etc.)
        return int(stage_str.split("_")[1])
    return int(stage_str)


def save_vae_metadata_to_pickle(
    output_dir: str | Path,
    train_stages: list[str],
    test_stages: list[str],
    library_sizes_dict: dict[str, Any],
    stage_min: int,
    stage_max: int
) -> dict[str, Any]:
    """
    Unified metadata saving for both scVI and MultiVI (publication-ready).

    Saves encoder-inferred library sizes to metadata.pkl for consistent generation.
    Uses global pool from training data only to avoid data leakage and support extrapolation.

    Args:
        output_dir: Directory to save metadata.pkl
        train_stages: list of training stage names
        test_stages: list of test stage names
        library_sizes_dict: Model-specific library size dictionaries (global pool)
            - For scVI: {'global_train_lib_sizes': tensor}
            - For MultiVI: {'rna_global_train_lib_sizes': tensor,
                           'atac_global_train_lib_sizes': tensor}
        stage_min: Minimum numeric stage value
        stage_max: Maximum numeric stage value

    Returns:
        dict: The saved metadata
    """
    import pickle
    from pathlib import Path

    output_dir = Path(output_dir)


    metadata = {
        "train": train_stages,
        "test": test_stages,
        "all": sorted(set(train_stages) | set(test_stages)),
        "stage_min": stage_min,
        "stage_max": stage_max,
        **library_sizes_dict,  # Unpack model-specific keys
    }


    metadata_path = output_dir / "metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    return metadata


def _get_stage_range(stages: list[str]) -> tuple[int, int]:
    """Get min and max numeric values from a list of stage strings."""
    numeric_values = [_extract_stage_number(s) for s in stages]
    if not numeric_values:
        raise ValueError("No stages provided - cannot compute range")
    return min(numeric_values), max(numeric_values)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value to use
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.benchmark = False
        th.backends.cudnn.deterministic = True


