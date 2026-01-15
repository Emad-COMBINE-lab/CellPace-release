"""Enhanced logging with Rich console + file persistence."""

import atexit
import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.theme import Theme

from cellpace.utils.common import is_main_process

colorblind_theme = Theme({
    "info": "cyan",
    "success": "bold cyan",
    "warning": "bold yellow",
    "error": "bold red",
    "progress": "magenta",
})

console = Console(theme=colorblind_theme)
_file_console = Console(file=None, force_terminal=False, no_color=True, width=120)
_log_file = None


def setup_logging(log_path: Path) -> None:
    global _log_file
    if _log_file is not None:
        _log_file.close()

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(log_path, 'w')

    # Register cleanup handler to ensure file is closed on exit
    atexit.register(close_logging)


def close_logging() -> None:
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None


def log_print(*args, **kwargs) -> None:
    if not is_main_process():
        return

    console.print(*args, **kwargs)

    if _log_file is not None:
        with _file_console.capture() as capture:
            _file_console.print(*args, **kwargs)
        _log_file.write(capture.get())
        _log_file.flush()


class MainProcessFilter(logging.Filter):
    """Only allow logging from main process in distributed training."""
    def filter(self, record):
        return is_main_process()


def setup_logger(name: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    console_handler = RichHandler(
        console=console,
        show_level=False,
        show_time=False,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        markup=True,
    )
    console_handler.setLevel(level)
    console_handler.addFilter(MainProcessFilter())
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def log_config_panel(title: str, config_dict: dict[str, str]) -> None:
    """Display configuration in a clean panel."""
    if not is_main_process():
        return
    lines = [f"{k:15s} {v}" for k, v in config_dict.items()]
    panel = Panel.fit("\n".join(lines), title=f"[bold]{title}[/bold]", border_style="blue")
    console.print(panel)


