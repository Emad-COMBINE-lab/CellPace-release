# CellPace: A temporal diffusion-forcing framework for simulation, interpolation and forecasting of single-cell dynamics

## 📋 System Requirements

### Software Dependencies
- **Python:** 3.12
- **PyTorch:** 2.5.1 with CUDA 11.8
- **Key packages:** scvi-tools 1.4.0, scanpy 1.11.5, pytorch-lightning 2.6.0
- See `environment.yaml` for complete list of dependencies

### Tested On
- **OS:** Ubuntu 20.04 (Linux 5.4.0)
- **GPU:** NVIDIA GeForce RTX 3090
- **CUDA:** 11.8 (via PyTorch)

## 🔧 Installation

### Option 1: Quick Install (Recommended)

```bash
bash install.sh
conda activate cellpace
```
**Typical install time:** ~3-5 minutes (tested on Ubuntu 20.04)

### Option 2: From Environment File

```bash
conda env create -f environment.yaml
conda activate cellpace
pip install -e .
```

## 📊 Data

CellPace requires temporal single-cell data in AnnData or MuData format.

### Input Format

**For scVI (RNA-only):**
- Format: `.h5ad` (AnnData)
- Required:
  - `adata.obs['stage_column']` - Temporal labels (e.g., timepoints, somite stages)
  - `adata.layers['raw_counts']` - Raw UMI counts

**For MultiVI (RNA+ATAC):**
- Format: `.h5mu` (MuData)
- Required:
  - `mdata['rna']` and `mdata['atac']` modalities
  - Both modalities share the same temporal column

### Example

```python
import scanpy as sc

# Load your data
adata = sc.read_h5ad("your_temporal_data.h5ad")

# Verify required fields
assert 'your_stage_column' in adata.obs.columns
assert 'raw_counts' in adata.layers.keys()
```

### Data

Demo data is provided in `demo/data/`:
- `demo_retinal_progenitor.h5ad` - Retinal progenitor cells, balanced subset with 50 cells per somite stage (~94MB, included in the repo)

Full data will be available on Zenodo:
- `prcd_retinal_progenitor.h5ad` - Full retinal progenitor dataset (~1.9GB, for full-scale experiments)

### Using Your Own Data

Replace the demo data path in the config file and ensure your AnnData has:
- `adata.layers['raw_counts']` - Raw UMI counts (not normalized or log-normalized)
- `adata.obs['your_stage_column']` - Temporal labels

Then update the config:
```yaml
data:
  paths:
    input: path/to/your_data.h5ad
  columns:
    stage_real: your_stage_column
```

## 🚀 Quick Start

CellPace uses a two-stage pipeline: (1) VAE training for latent representation, (2) Diffusion model training for temporal dynamics.

**Demo dataset**: The retinal progenitor demo is subsampled to ~50 cells per stage (1,550 total cells) for quick testing.

### Stage 1: Train VAE (scVI/MultiVI)

```bash
# For RNA-only data
cellpace train --model scvi --config-file demo/configs/retinal.yaml

# For multiome (RNA+ATAC) data, need to download the data separately
cellpace train --model multivi --config-file demo/configs/multiome.yaml
```

**Expected runtime**: ~30 seconds (demo data with early stopping, NVIDIA RTX 3090)

**Expected output**:
```
demo/output/retinal/scvi/
├── model.pt                 # Trained scVI model weights (~8.7MB)
└── metadata.pkl             # Model metadata
```

### Stage 2: Train Diffusion Model

```bash
# Use 2 GPUs for faster training
CUDA_VISIBLE_DEVICES=0,1 cellpace train --model dif --config-file demo/configs/retinal.yaml
```

**Expected runtime**: ~15 minutes for 10K steps (demo data, 2× NVIDIA RTX 3090)

**Expected output**:
```
demo/output/retinal/dif-pred_x0/
├── checkpoints/
│   ├── checkpoint_step=10000.ckpt   # Model checkpoint (~22MB)
│   └── last.ckpt                    # Latest checkpoint
└── input_seq/                       # Cached training sequences
```

### Stage 3: Generation

```bash
cellpace generate --model dif --config-file demo/configs/retinal.yaml
```

**Expected runtime**: ~2 minute (demo data, NVIDIA RTX 3090)

**Expected output**:
```
demo/output/retinal/dif-pred_x0/infer_ns_step_10000/
├── generated_latents.h5ad   # Generated latent representations (~32MB)
├── generated_gex.h5ad       # Generated gene expression (~14MB)
├── real_latents.h5ad        # Real latents for comparison
└── real_gex.h5ad            # Real gene expression for comparison
```

## 📄 License

This software is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the `LICENSE` file for details.

## 🙏 Acknowledgments

CellPace builds upon and acknowledges the following open-source projects and datasets:

### Core Dependencies

**scVI-tools**
- VAE encoder/decoder (Stage 1) uses scVI and MultiVI models
- GitHub: [scverse/scvi-tools](https://github.com/scverse/scvi-tools)
- Paper: scvi-tools: A library for deep probabilistic analysis of single-cell omics data

**Diffusion Forcing Framework**
- Our diffusion model (Stage 2) is adapted from the Diffusion Forcing framework
- GitHub: [buoyancy99/diffusion-forcing](https://github.com/buoyancy99/diffusion-forcing)
- Paper: Chen, B., et al., Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 2024. 37: p. 24081-24125.

### Datasets

**Mouse Embryo Dataset**
- Source: [CellxGene](https://cellxgene.cziscience.com/collections/45d5d2c3-bc28-4814-aed6-0bb6f0e11c82)
- Paper: Qiu, C., et al., A single-cell time-lapse of mouse prenatal development from gastrula to birth. Nature, 2024. 626(8001): p. 1084-1093.

**Mouse Palate Development Multiome Dataset**
- Source: [fangfang0906/Single_cell_multiome_palate](https://github.com/fangfang0906/Single_cell_multiome_palate)
- Paper: Yan, F., et al., Single-cell multiomics decodes regulatory programs for mouse secondary palate development. Nature communications, 2024. 15(1): p. 821.

### Benchmark Methods

We compare against the following methods for temporal single-cell generation (sorted by A-Z):

1. [cfDiffusion](https://github.com/SuperheroBetter/cfDiffusion): Zhang, T., et al., cfDiffusion: diffusion-based efficient generation of high quality scRNA-seq data with classifier-free guidance. Briefings in Bioinformatics, 2025. 26(1): p. bbaf071.
2. [CFGen](https://github.com/theislab/CFGen): Palma, A., et al., Generating multi-modal and multi-attribute single-cell counts with CFGen. arXiv preprint arXiv:2407.11734, 2024.
3. [ESCFD](https://github.com/JayYang133/ESCFD): Li, S., et al. ESCFD: Probabilistic Flow Diffusion Model for Accelerated High-Quality Single-Cell RNA-seq Data Synthesis. in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 2025.
4. [scDiffusion](https://github.com/EperLuo/scDiffusion): Luo, E., et al., scDiffusion: conditional generation of high-quality single-cell data using diffusion model. Bioinformatics, 2024. 40(9): p. btae518.
5. [scIMF](https://github.com/QiJiang-QJ/scIMF): Jiang, Q., et al., Learning collective multi-cellular dynamics from temporal scRNA-seq via a transformer-enhanced Neural SDE. arXiv preprint arXiv:2505.16492, 2025.
6. [scNODE](https://github.com/rsinghlab/scNODE): Zhang, J., et al., scNODE: generative model for temporal single cell transcriptomic data prediction. Bioinformatics, 2024. 40(Supplement_2): p. ii146-ii154.

We thank the authors of these methods for making their code publicly available.


