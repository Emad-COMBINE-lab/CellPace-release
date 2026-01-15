"""CellPace: Temporal single-cell generation via latent diffusion."""

# Core models
# Data management
from .data import DataManager
from .model.dif.core import DiffusionForcing
from .model.dif.inference import DiFInference
from .model.dif.utils import GenerationConfig, GenerationData, ModelConfig
from .model.vae.base import VAEUtils
from .model.vae.multivi import train_multivi

# VAE/scVI components
from .model.vae.scvi import SCVIModel, train_scvi
from .utils.common import is_main_process, set_random_seeds

# Utilities
from .utils.logging import close_logging, log_print, setup_logging

__version__ = "3.0.0"

__all__ = [
    # Core DiF
    "DiffusionForcing",
    "DiFInference",
    "GenerationConfig",
    "ModelConfig",
    "GenerationData",
    # VAE/scVI
    "SCVIModel",
    "train_scvi",
    # VAE/MultiVI
    "train_multivi",
    # VAE Base
    "VAEUtils",
    # Data
    "DataManager",
    # Utils
    "setup_logging",
    "log_print",
    "close_logging",
    "set_random_seeds",
    "is_main_process",
]
