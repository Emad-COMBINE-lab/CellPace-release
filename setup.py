"""
Setup configuration for CellPace package.

cmd: pip install -e . --no-deps
"""

import os
from setuptools import setup, find_packages

setup(
    name="cellpace",
    version="1.0",
    author="TheChenSu",
    author_email="",
    description="CellPace: A temporal diffusion-forcing framework for simulation, interpolation and forecasting of single-cell dynamics",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(include=["cellpace", "cellpace.*"]),
    # Dependencies managed by install.sh - see that file for installation
    install_requires=[],
    entry_points={
        "console_scripts": [
            "cellpace=cellpace.__main__:main",
        ],
    },
    python_requires=">=3.12,<=3.14",
)
