#!/bin/bash
# CellPace Installation Script for Linux

set -e

echo "========================================"
echo "CellPace Installation (Linux)"
echo "========================================"

if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

ENV_NAME="cellpace"

echo ""
echo "Step 1: Setting up conda environment '$ENV_NAME'..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "  Environment '$ENV_NAME' already exists. Removing and recreating..."
    conda env remove -n $ENV_NAME -y
fi
echo "  Creating environment with Python 3.12..."
conda create -n $ENV_NAME python=3.12 -y

echo ""
echo "Step 2: Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo ""
echo "Step 3: Installing PyTorch with CUDA support..."
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo ""
echo "Step 4: Installing core scientific packages..."
conda install -c conda-forge numpy pandas scipy matplotlib seaborn scikit-learn -y

echo ""
echo "Step 5: Installing single-cell analysis packages..."
pip install scanpy scvi-tools anndata mudata

echo ""
echo "Step 6: Installing additional dependencies..."
pip install pytorch-lightning einops omegaconf wandb tqdm umap-learn POT rich-argparse openpyxl

echo ""
echo "Step 7: Installing testing framework..."
pip install pytest pytest-cov
pip install ruff

echo ""
echo "Step 8: Installing CellPace..."
pip install -e .

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "To use CellPace:"
echo "  conda activate $ENV_NAME"
echo "  cellpace --help"
echo ""
echo "To run tests:"
echo "  pytest tests/ -v"
echo ""
