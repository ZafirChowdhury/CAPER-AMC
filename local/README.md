# CAPER-AMC

**C**onditional **A**daptive **P**ath **E**xpert **R**outing for **A**utomatic **M**odulation **C**lassification

A VRAM-optimized deep learning model for radio signal modulation classification on the [RML2016.10a](https://www.deepsig.ai/datasets) dataset. CAPER-AMC uses a mixture-of-experts transformer architecture with SNR-aware conditional routing to achieve high accuracy across a wide SNR range (−6 dB to +18 dB).

---

## Features

- **Conditional Top-1 Routing** — dynamically routes each sample through one of three attention paths (full, causal, or local-window) based on learned signal characteristics
- **Physics-Informed Front-End** — computes amplitude, phase, and delta features from raw IQ samples before the transformer stack
- **Spectral Context** — appends FFT magnitude features to enrich the input representation
- **QAM & Analog Expert Heads** — dedicated refinement heads for QAM and analog modulation families
- **SNR-Weighted Loss** — applies higher emphasis to low-SNR samples during training
- **Supervised Contrastive Learning** — optional inter-class separation objective
- **Exponential Moving Average (EMA)** — maintains a smoothed model for checkpointing and evaluation
- **Mixed-Precision (AMP) Training** — full FP16 autocast + gradient scaling on CUDA
- **All data on VRAM** — the entire dataset is loaded into GPU memory for maximum throughput

---

## Repository Structure

```
.
├── caper_amc_local.py            # Core model, training loop, and VRAM data loader
├── CAPER_AMC_All_Figures.py      # Self-contained training + figure generation script
├── caper_amc_figure_generator.py # Standalone figure generator (loads saved checkpoint)
├── plot_results.py               # Quick post-training evaluation plots
├── requirements.txt
└── README.md
```

### Script Overview

| Script | Purpose |
|---|---|
| `caper_amc_local.py` | **Primary training script.** Defines `CAPERAMC`, all training/eval loops, and the VRAM data loader. Run this to train the model. |
| `CAPER_AMC_All_Figures.py` | All-in-one script: trains the model *and* generates all publication figures in a single run. |
| `caper_amc_figure_generator.py` | Generates full paper figure suite from an existing checkpoint without retraining. |
| `plot_results.py` | Lightweight script that generates a confusion matrix and SNR accuracy curve from a saved checkpoint. |

---

## Installation

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended. The model trains on CPU but is significantly slower.

> **PyTorch note:** Install the CUDA-enabled build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt` if your environment does not already have it.

---

## Dataset

Download the **RML2016.10a** dataset (`.pkl` / `.pickle`) from [DeepSig](https://www.deepsig.ai/datasets) and place it in the project root. A `.zip` containing the pickle file is also supported and will be extracted automatically.

```
.
├── RML2016.10a_dict.pkl   ← place here
└── ...
```

---

## Usage

### Train the model

```bash
python caper_amc_local.py
```

Outputs are saved to `./caper_amc_results/`:

- `caper_amc_best_vram.pth` — best EMA checkpoint (saved when validation score improves)

### Train and generate all figures

```bash
python CAPER_AMC_All_Figures.py
```

Generates the checkpoint *and* all publication-quality figures in one run.

### Generate figures from a saved checkpoint

```bash
python caper_amc_figure_generator.py
```

With explicit paths:

```bash
python caper_amc_figure_generator.py \
    --training-script caper_amc_local.py \
    --checkpoint caper_amc_results/caper_amc_best_vram.pth \
    --dataset RML2016.10a_dict.pkl \
    --output-dir caper_amc_results/paper_figures
```

With optional training history curves:

```bash
python caper_amc_figure_generator.py \
    --history caper_amc_results/training_history.csv
```

> **Note:** The training script prints per-epoch metrics but does not save a history CSV automatically. Post-training figures (confusion matrix, SNR curves, routing analysis, calibration) are always generated from the checkpoint. Training curves require a separately saved CSV.

#### `caper_amc_figure_generator.py` arguments

| Argument | Default | Description |
|---|---|---|
| `--training-script` | `caper_amc_local.py` | Path to the training Python file |
| `--checkpoint` | `caper_amc_results/caper_amc_best_vram.pth` | Path to saved EMA checkpoint |
| `--dataset` | *(auto-detect)* | Path to `.pkl`, `.pickle`, or `.zip` dataset |
| `--output-dir` | `caper_amc_results/paper_figures` | Output directory for figures and tables |
| `--history` | *(auto-detect)* | Optional `training_history.csv` for training curves |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--batch-size` | `1024` | Evaluation batch size |
| `--latency-batch-size` | `512` | Batch size for latency benchmarking |
| `--latency-repeats` | `40` | Timed repetitions per routing mode |
| `--skip-route-benchmark` | `false` | Skip per-route accuracy and latency evaluation |
| `--no-pdf` | `false` | Save PNG only instead of PNG + PDF |

### Quick evaluation plots

```bash
python plot_results.py
```

Requires an existing checkpoint at `./caper_amc_results/caper_amc_best_vram.pth`. Saves:

- `caper_amc_results/confusion_matrix.png`
- `caper_amc_results/snr_accuracy_curve.png`

---

## Model Configuration

Key hyperparameters are defined at the top of `caper_amc_local.py`:

| Parameter | Value | Description |
|---|---|---|
| `D_MODEL` | 128 | Transformer hidden dimension |
| `N_HEADS` | 4 | Attention heads |
| `ATTENTION_LAYERS_PER_ROUTE` | 2 | Transformer layers per routing path |
| `D_FF` | 384 | Feed-forward expansion dimension |
| `DROPOUT` | 0.15 | Dropout rate |
| `LOCAL_WINDOW` | 9 | Local attention window size |
| `ROUTING_MODE` | `conditional_top1` | Routing strategy |
| `ROUTER_TEMP_START` | 2.0 | Initial Gumbel softmax temperature |
| `ROUTER_TEMP_END` | 0.55 | Final router temperature |

### Training hyperparameters

| Parameter | Value |
|---|---|
| `BATCH_SIZE` | 1024 |
| `NUM_EPOCHS` | 100 |
| `WARMUP_EPOCHS` | 5 |
| `LEARNING_RATE` | 3e-4 |
| `WEIGHT_DECAY` | 1e-4 |
| `EMA_DECAY` | 0.999 |
| `GRAD_CLIP_NORM` | 1.0 |

### SNR range & data splits

| Setting | Value |
|---|---|
| `MIN_SNR` | −6 dB |
| `MAX_SNR` | +18 dB |
| Train / Val / Test | 80 / 10 / 10 % |

---

## Output Files

### Training (`caper_amc_results/`)

| File | Description |
|---|---|
| `caper_amc_best_vram.pth` | Best EMA model checkpoint |

### Figure generator (`caper_amc_results/paper_figures/`)

#### Figures

| File | Description |
|---|---|
| `fig01_architecture.*` | Model architecture diagram |
| `fig02_input_signal.*` | Example IQ waveforms per modulation |
| `fig03_confusion_matrix.*` | Normalized confusion matrix |
| `fig04_per_class_metrics.*` | Per-class precision, recall, F1 |
| `fig05_performance_vs_snr.*` | Accuracy and F1 vs SNR |
| `fig06_routing_vs_snr.*` | Routing distribution vs SNR |
| `fig07_routing_by_class.*` | Routing distribution by modulation |
| `fig08_calibration.*` | Reliability / calibration diagram |
| `fig09_snr_estimation.*` | SNR estimation scatter and error |
| `fig10_uncertainty_vs_snr.*` | Prediction entropy vs SNR |
| `fig11_route_tradeoff.*` | Accuracy–latency trade-off per route |
| `fig12_training_curves.*` | Train/val accuracy and loss curves (requires history CSV) |

#### Tables (CSV / JSON)

| File | Description |
|---|---|
| `test_predictions.csv` | Per-sample targets, predictions, SNR, routing |
| `per_class_metrics.csv` | Precision, recall, F1 per modulation class |
| `classification_report.csv` | Full sklearn classification report |
| `confusion_matrix_counts.csv` | Raw confusion matrix |
| `confusion_matrix_normalized.csv` | Row-normalized confusion matrix |
| `performance_by_snr.csv` | Accuracy and F1 at each SNR level |
| `routing_by_snr.csv` | Route assignment fractions per SNR |
| `routing_by_modulation.csv` | Route assignment fractions per class |
| `uncertainty_by_snr.csv` | Mean prediction entropy per SNR |
| `snr_estimation_by_snr.csv` | SNR estimation MAE per true SNR |
| `calibration_bins.csv` | Confidence calibration bin data |
| `route_accuracy_latency_benchmark.csv` | Per-route accuracy and inference latency |
| `summary_metrics.json` | Overall accuracy, Macro-F1, ECE, SNR MAE |

---

## Hardware Requirements

| Component | Recommended |
|---|---|
| GPU | NVIDIA RTX 3060 or better (≥ 12 GB VRAM) |
| RAM | ≥ 16 GB system RAM |
| CUDA | 11.8+ |
| Python | 3.9+ |

The entire RML2016.10a dataset (~450 MB) is loaded into GPU memory at the start of training. A GPU with at least 8 GB VRAM can run with a reduced `BATCH_SIZE`.
