"""Argument parser for CellPace CLI."""

import argparse

from rich_argparse import RichHelpFormatter

from cellpace import __version__


def setup_argparse() -> argparse.ArgumentParser:
    """Setup argument parser for CellPace CLI."""

    class CustomRichHelpFormatter(RichHelpFormatter):
        """Custom formatter that preserves epilog formatting."""
        def _split_lines(self, text, width):
            """Preserve line breaks in epilog."""
            return text.splitlines()

    parser = argparse.ArgumentParser(
        prog="cellpace",
        description="CellPace - Spatial-Temporal Single-Cell Generation via Latent Diffusion Forcing",
        epilog="""
Requirements:
  • CUDA-capable GPU required
  • Python 3.12+
  • See installation guide for dependencies

Examples:
  Train scVI encoder:
    $ cellpace train --model scvi --config-file demo/configs/retinal.yaml

  Train MultiVI encoder (for multimodal RNA+ATAC data):
    $ cellpace train --model multivi --config-file demo/configs/retinal.yaml

  Train diffusion model (requires trained scVI/MultiVI):
    $ cellpace train --model dif --config-file demo/configs/retinal.yaml

  Generate cells:
    $ cellpace generate --model dif --config-file demo/configs/retinal.yaml

  With config overrides:
    $ cellpace train --model dif --config-file demo/configs/retinal.yaml \\
        --config dif.training.max_steps=50000 scvi.n_latent=128
""",
        formatter_class=CustomRichHelpFormatter,
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "command",
        choices=["train", "generate"],
        help="train - Train VAE or diffusion model; generate - Generate synthetic cells",
    )
    required.add_argument(
        "--model",
        required=True,
        choices=["scvi", "multivi", "dif"],
        help="Model type: scvi (VAE for RNA-only), multivi (VAE for multimodal RNA+ATAC), dif (diffusion model)",
    )

    # Configuration
    config_group = parser.add_argument_group("configuration")
    config_group.add_argument(
        "--config-file",
        default="demo/configs/retinal.yaml",
        metavar="PATH",
        help="Path to YAML configuration file (default: demo/configs/retinal.yaml)",
    )
    config_group.add_argument(
        "--config",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config parameters (e.g., dif.training.max_steps=50000)",
    )

    # Generation options
    gen_group = parser.add_argument_group("generation options")
    gen_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to trained model checkpoint (auto-detected if not provided)",
    )

    # Other options
    parser.add_argument(
        "--version", action="version", version=f"CellPace {__version__}"
    )

    return parser
