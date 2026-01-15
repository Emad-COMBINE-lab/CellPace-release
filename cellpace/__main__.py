"""CellPace CLI entry point.

Usage:
    cellpace train --model scvi --config-file demo/configs/retinal.yaml
    cellpace train --model dif --config-file demo/configs/retinal.yaml
    cellpace generate --model dif --config-file demo/configs/retinal.yaml
"""

import logging as python_logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
python_logging.getLogger("wandb").setLevel(python_logging.ERROR)
python_logging.getLogger("pytorch_lightning").setLevel(python_logging.ERROR)

from pathlib import Path

import torch
from omegaconf import OmegaConf

from cellpace.cli.args import setup_argparse
from cellpace.cli.commands import generate, train
from cellpace.data.validator import DataValidator
from cellpace.utils.common import set_random_seeds
from cellpace.utils.logging import log_print


def main() -> None:
    """Main entry point for CellPace CLI."""
    parser = setup_argparse()
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. No GPU detected.")
    torch.set_float32_matmul_precision("high")

    config_file = args.config_file
    if not Path(config_file).is_absolute() and not Path(config_file).exists():
        config_file = str(Path.cwd() / config_file)
    config = OmegaConf.load(config_file)

    if args.config:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.config))

    DataValidator.validate_config(config, args.model)

    seed = config.experiment.seed
    set_random_seeds(seed)
    log_print(f"Set random seed to {seed}")

    if args.command == "train":
        train(config, args.model)
    elif args.command == "generate":
        generate(config, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
