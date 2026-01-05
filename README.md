# OptiCPL - Two-Stage Pipeline

Predict nanomaterial fabrication parameters from gluminescence data via a two-stage deep learning pipeline.

**Pipeline**: `glum (40-dim) → Stage2 → Features (200-dim) → Stage1 → Fabrication Parameters (8-dim)`

## Quick Start

### 1. Install Dependencies

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn openpyxl
```
### 2. Data and Pretrained Model install

https://huggingface.co/Zylqaq/OptiCPL
```
OptiCPL
├── Checkpoints
├── OptiCPL_refactored
├── Data
│   └── SEM_ALL
│   └── Data_complete_corrected_v7.xlsx
│   └── DF_IMG.xlsx
└── main.py
```

### 2. Basic Usage

#### Mode 1: Train Stage1 Model

Train the Stage1 model on pre-computed features:

```bash
# Basic training
python main.py train

# With experiment tracking (recommended)
python main.py train --experiment-name "my_experiment"

# Custom hyperparameters
python main.py train --epochs 2000 --lr 0.001 --batch-size 64 --device cuda
```

**Training data**: `/home/ylzhang/Nano_HF/OptiCPL/Data/DF_IMG.xlsx` (200-dim features + 8 targets)

#### Mode 2: End-to-End Pipeline (from glum)

Run complete pipeline from raw glum data to fabrication parameters:

```bash
# Default glum data
python main.py pipeline --experiment-name "pipeline_run"

# Custom glum data
python main.py pipeline --glum-data /path/to/glum.xlsx --experiment-name "batch_1"
```

**Input**: Excel/CSV with 40-dim glum features (column: "glum")
**Output**: 8 fabrication parameters in `Training_Logs/*/results/predictions.csv`

#### Mode 3: Raw Data Pipeline

**Feature**: Process raw data with spectral measurements and SEM images:

```bash
# Process from original data file with spectra and images
python main.py raw-pipeline --experiment-name "raw_data_run"
```

**Pipeline**:
```
Data_complete_corrected_v7.xlsx
├── Spectra (PL, EX_sample, CD_sample, CD_template, EX_template)
│   └── VAE Model → Optical Features (32-dim)
├── glum (40-dim)
│   └── Pass-through
└── SEM Images (x_1.tif, x_3.tif, x_5.tif)
    └── SimCLR Model → IMG Features (128-dim)
```

#### Mode 4: Predict with Trained Model

Make predictions using pre-computed features and trained Stage1 model:

```bash
# Default settings
python main.py predict

# Custom paths
python main.py predict --input features.csv --output predictions.csv --model stage1_model.pth --device cpu
```

**Input**: CSV with 200-dim features [IMG(128) + glum(40) + Optical(32)]
**Output**: 8 fabrication parameters

## Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `mode` | Operation mode: `train`, `pipeline`, `raw-pipeline`, `predict` | Required |
| `--experiment-name` | Name for organized experiment tracking | None |
| `--glum-data` | Input glum file (pipeline mode) | See config |
| `--model` | Stage1 model checkpoint path (predict mode) | See config |
| `--input` | Input features CSV (predict mode) | See config |
| `--output` | Output predictions path (predict mode) | See config |
| `--epochs` | Training epochs | 4000 |
| `--lr` | Learning rate | 0.0005 |
| `--batch-size` | Batch size | 32 |
| `--device` | Device: auto/cpu/cuda | auto |
| `--raw-data` | Input raw data file (raw-pipeline mode) | `/home/ylzhang/Nano_HF/OptiCPL/Data/Data_complete_corrected_v7.xlsx` |
| `--img-dir` | SEM images directory (raw-pipeline mode) | `/home/ylzhang/Nano_HF/SunHaifeng/Data/SEM_ALL/` |
| `--vae-model` | VAE model path (raw-pipeline mode) | `/home/ylzhang/Nano_HF/OptiCPL/Checkpoints/Pretrained/Spec/Spec_model.pth` |
| `--img-model` | SimCLR model path (raw-pipeline mode) | `/home/ylzhang/Nano_HF/OptiCPL/Checkpoints/Pretrained/IMG/trained_resnet18_simclr2.pth` |

## Data Formats

### Input Formats

**Training (Excel)**:
- 200-dim features: IMG (128) + glum (40) + Optical (32)
- 8 targets: Deposition_angle, Deposition_speed, β, pitch, n, CsBr, PbBr2, Soaking_time
- **File**: `/home/ylzhang/Nano_HF/OptiCPL/Data/DF_IMG.xlsx`

**Raw Data Pipeline**:
- **Raw data file**: `/home/ylzhang/Nano_HF/OptiCPL/Data/Data_complete_corrected_v7.xlsx`
- Contains 5 spectral columns: PL(40), EX_sample(200), CD_sample(200), CD_template(200), EX_template(200)
- glum column (40-dim)
- 8 fabrication parameters (targets)
- SEM images: three images from different angles in `/home/ylzhang/Nano_HF/SunHaifeng/Data/SEM_ALL/`

**Prediction (CSV)**:
- Pre-computed 200-dim features [IMG + glum + Optical]

### Outputs

**Experiment Tracking** (`Training_Logs/YYYYMMDD_HHMMSS_experiment_name/`):
```
├── raw_data/            # Copy of input raw data (raw-pipeline mode)
├── features/
│   ├── glum_parsed.csv      # Parsed glum data
│   ├── optical_features.csv # 32-dim Optical features
│   ├── img_features.csv     # 128-dim IMG features
│   └── combined_features.csv # 200-dim merged features
├── models/
│   ├── stage1_best_model.pth
│   └── stage2/              # Copies of VAE/SimCLR models
├── scalers/             # StandardScaler objects
├── results/             # predictions.csv (8 fabrication parameters)
├── visualizations/      # Training curves and prediction plots
├── metadata.json        # Processing metadata (raw-pipeline only)
└── training_info.txt    # Complete configuration
```

**Raw-pipeline mode** additionally saves all intermediate feature files for debugging and reproducibility.
