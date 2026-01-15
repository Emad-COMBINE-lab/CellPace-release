"""Configuration management utilities for reproducible experiments."""

import subprocess


def get_git_info() -> str:
    """Get current git commit hash and dirty status."""
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        git_dirty = subprocess.check_output(['git', 'status', '--porcelain']).decode('ascii').strip()
        if git_dirty:
            git_hash += " (dirty)"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        git_hash = "unknown"
    return git_hash


