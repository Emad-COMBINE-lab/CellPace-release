"""Data validation utilities for CellPace."""

# Standard library
import pickle
from pathlib import Path
from typing import Any

# Third-party
import scanpy as sc


class DataValidator:
    """Collection of validation methods for data processing."""

    @staticmethod
    def validate_stage_splits(stage_splits: dict[str, list[str | None]]) -> None:
        """
        Validate that stage splits dictionary has required structure.

        Args:
            stage_splits: Dictionary with 'train' key (required) and optional val/test keys

        Raises:
            ValueError: If stage_splits is None or missing 'train' key
        """
        if stage_splits is None or "train" not in stage_splits:
            raise ValueError("stage_splits must be provided with at least 'train' key")

        if not isinstance(stage_splits["train"], list) or len(stage_splits["train"]) == 0:
            raise ValueError("stage_splits['train'] must be a non-empty list")

    @staticmethod
    def validate_adata_fields(
        adata: sc.AnnData, required_layers: list[str], required_obs: list[str]
    ) -> None:
        """
        Validate that AnnData has required layers and obs columns.

        Args:
            adata: AnnData object to validate
            required_layers: list of required layer names
            required_obs: list of required obs column names

        Raises:
            ValueError: If required fields are missing
        """
        missing_layers = [layer for layer in required_layers if layer not in adata.layers]
        missing_obs = [col for col in required_obs if col not in adata.obs]

        if missing_layers or missing_obs:
            raise ValueError(
                f"Required fields missing:\n"
                f"  Layers: {missing_layers if missing_layers else 'OK'}\n"
                f"  Obs columns: {missing_obs if missing_obs else 'OK'}"
            )

    @staticmethod
    def validate_normalization_params(
        stage_min: int | None, stage_max: int | None
    ) -> None:
        """
        Validate stage normalization parameters.

        Args:
            stage_min: Minimum stage value
            stage_max: Maximum stage value

        Raises:
            ValueError: If parameters are None or equal
        """
        if stage_min is None or stage_max is None:
            raise ValueError(
                "stage_min and stage_max must be provided from DataManager"
            )

        if stage_max == stage_min:
            raise ValueError(
                f"Dataset contains only one unique stage value ({stage_min}) - "
                "cannot normalize temporal range. Need at least 2 different stages for training."
            )

    @staticmethod
    def validate_cache_exists(cache_paths: list[Path]) -> bool:
        """
        Check if all cache files exist.

        Args:
            cache_paths: list of cache file paths to check

        Returns:
            True if all files exist, False otherwise
        """
        return all(f.exists() for f in cache_paths)

    @staticmethod
    def validate_cache_metadata(
        metadata_path: Path, expected_params: dict[str, Any]
    ) -> bool:
        """
        Validate cache metadata matches expected parameters.

        Args:
            metadata_path: Path to metadata pickle file
            expected_params: Dictionary of param_name: expected_value

        Returns:
            True if metadata is valid and matches, False otherwise
        """
        if not metadata_path.exists():
            return False

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Check all expected parameters
        for param_name, expected_value in expected_params.items():
            if metadata.get(param_name) != expected_value:
                return False

        return True

    @staticmethod
    def validate_vae_hash_consistency(
        cached_hash: str | None, current_hash: str | None, cache_path: Path
    ) -> None:
        """
        Validate that cached scVI model hash matches current model.

        Args:
            cached_hash: Hash stored in cache
            current_hash: Current model hash
            cache_path: Path to cache for error message

        Raises:
            ValueError: If hashes don't match or are missing
        """
        # Both None means no scVI tracking - this is valid
        if cached_hash is None and current_hash is None:
            return

        # One is None but not the other - incompatible
        if cached_hash is None:
            raise ValueError(
                f"Cache file {cache_path} doesn't track scVI model but current config requires it. "
                "Delete the cache or set force_regenerate=true"
            )

        if current_hash is None:
            raise ValueError(
                f"Cache file {cache_path} tracks scVI model but current config doesn't. "
                "Delete the cache or set force_regenerate=true"
            )

        # Both are not None - they must match
        if cached_hash != current_hash:
            raise ValueError(
                f"scVI model has changed! Cached: {cached_hash}, Current: {current_hash}. "
                f"Delete {cache_path} or set force_regenerate=true"
            )

    @staticmethod
    def validate_normalization_consistency(
        cached_min: int,
        cached_max: int,
        expected_min: int,
        expected_max: int,
        cache_path: Path,
    ) -> None:
        """
        Validate that cached normalization matches expected parameters.

        Args:
            cached_min: Minimum from cache
            cached_max: Maximum from cache
            expected_min: Expected minimum
            expected_max: Expected maximum
            cache_path: Path to cache for error message

        Raises:
            ValueError: If normalization parameters don't match
        """
        if cached_min != expected_min or cached_max != expected_max:
            raise ValueError(
                f"Cached normalization [{cached_min}, {cached_max}] doesn't match "
                f"current training normalization [{expected_min}, {expected_max}]. "
                f"Delete {cache_path} or set force_regenerate=true"
            )

    @staticmethod
    def validate_cache_has_field(data: dict, field_name: str, cache_path: Path) -> None:
        """
        Validate that cached data has required field.

        Args:
            data: Loaded cache dictionary
            field_name: Required field name
            cache_path: Path to cache for error message

        Raises:
            ValueError: If field is missing
        """
        if field_name not in data:
            raise ValueError(
                f"Cache file {cache_path} is missing required field '{field_name}'. "
                "Delete the cache or set force_regenerate=true"
            )

    @staticmethod
    def validate_strategy(strategy: str, valid_strategies: list[str]) -> None:
        """
        Validate split strategy is recognized.

        Args:
            strategy: Strategy name from config
            valid_strategies: list of valid strategy names

        Raises:
            ValueError: If strategy is not recognized
        """
        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown split strategy: {strategy}. "
                f"Valid options: {', '.join(valid_strategies)}"
            )

    @staticmethod
    def validate_vae_type_consistency(
        vae_type: str,
        has_multiome_data: bool,
        data_path: Path
    ) -> None:
        """
        Validate VAE type is compatible with data format.

        Args:
            vae_type: VAE model type ('scvi' or 'multivi')
            has_multiome_data: Whether multiome (.h5mu) data was loaded
            data_path: Path to data file for error message

        Raises:
            ValueError: If MultiVI requested but data is not multiome
        """
        if vae_type == "multivi" and not has_multiome_data:
            raise ValueError(
                f"experiment.vae_to_use='multivi' requires multiome data (.h5mu file). "
                f"Current file: {data_path}"
            )

    @staticmethod
    def validate_mudata_time_labels(
        mdata,
        time_key: str,
        modalities: list[str | None] = None
    ) -> None:
        """
        Validate that time labels exist and are consistent across MuData modalities.

        For fully paired multiome data, ensures RNA and ATAC have identical time labels
        for each cell. This is critical for proper train/val/test splitting.

        Args:
            mdata: MuData object with multiple modalities
            time_key: Column name for time/stage labels (e.g., 'palantir_discrete')
            modalities: list of modality names to validate (default: ['rna', 'atac'])

        Raises:
            ValueError: If time labels are missing or inconsistent across modalities
        """
        import numpy as np

        if modalities is None:
            modalities = ['rna', 'atac']

        # Check that all modalities exist
        for mod in modalities:
            if mod not in mdata.mod:
                raise ValueError(
                    f"Expected modality '{mod}' not found in MuData. "
                    f"Available modalities: {list(mdata.mod.keys())}"
                )

        # Check that time_key exists in all modalities
        for mod in modalities:
            if time_key not in mdata.mod[mod].obs.columns:
                raise ValueError(
                    f"Time key '{time_key}' not found in {mod.upper()} modality. "
                    f"Available columns: {list(mdata.mod[mod].obs.columns)}"
                )

        # For paired data, validate that all modalities have identical time labels
        if len(modalities) > 1:
            reference_labels = mdata.mod[modalities[0]].obs[time_key].values

            for mod in modalities[1:]:
                mod_labels = mdata.mod[mod].obs[time_key].values

                if not np.array_equal(reference_labels, mod_labels):
                    n_mismatch = (reference_labels != mod_labels).sum()
                    raise ValueError(
                        f"Time labels mismatch between {modalities[0].upper()} and {mod.upper()}! "
                        f"{n_mismatch}/{len(reference_labels)} cells have different labels. "
                        f"Multiome data must have consistent metadata across modalities. "
                        f"Check your data preprocessing."
                    )

    @staticmethod
    def validate_config(config, model_type: str) -> None:
        """Validate config has required sections for model type and vae_to_use."""
        required = ["experiment", "data"]

        if model_type == "dif":
            required.append("dif")
            vae_type = config.experiment.get("vae_to_use", "scvi")
            required.append(vae_type)
        elif model_type in ["scvi", "multivi"]:
            required.append(model_type)

        missing = [s for s in required if s not in config]
        if missing:
            raise ValueError(f"Config missing required sections: {missing}")

