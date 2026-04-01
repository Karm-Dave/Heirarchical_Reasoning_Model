# GM-HRM Experiment Guide

This guide explains how to run all experiments for the **Gated-Memory Hierarchical Reasoning Model (GM-HRM)** and save results properly.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Configuration Files](#configuration-files)
4. [Running Experiments](#running-experiments)
5. [What to Change Before Running](#what-to-change-before-running)
6. [Output and Results](#output-and-results)
7. [Individual Scripts](#individual-scripts)
8. [Benchmarks and Ablations](#benchmarks-and-ablations)
9. [Quick Reference Commands](#quick-reference-commands)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Python Environment

Create and activate a virtual environment:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `torch>=2.10.0` (with CUDA support recommended)
- `numpy>=2.2.6`
- `pyyaml>=6.0.3`
- `pydantic`
- `argdantic`
- `huggingface_hub` (for downloading benchmark datasets)
- `matplotlib` (optional, for plot generation)
- `tqdm` (for progress bars)

### 3. Hardware Requirements

- **GPU (recommended):** CUDA-compatible GPU with at least 8GB VRAM
- **CPU fallback:** Will work but significantly slower
- **RAM:** At least 16GB recommended for larger benchmarks

---

## Project Structure

```
Heirarchical_Reasoning_Model/
├── configs/                 # Configuration YAML files
│   ├── main.yaml           # Main experiment configuration
│   └── smoke.yaml          # Quick smoke test configuration
├── data/                   # Data generators and datasets
│   ├── benchmarks/         # Generated benchmark datasets
│   ├── sudoku_generator.py # Sudoku data generator
│   ├── maze_generator.py   # Maze data generator
│   ├── math_generator.py   # Math chains generator
│   ├── mock_generator.py   # Mock data generator (for ARC, etc.)
│   └── datasets.py         # PyTorch dataset classes
├── models/                 # Model architecture
│   ├── gm_hrm.py          # Main GM-HRM model
│   ├── halting.py         # ACT halting mechanism
│   ├── memory.py          # External memory module
│   ├── fusion.py          # Gated fusion module
│   └── hrm_modules.py     # High/Low level reasoning modules
├── scripts/               # Executable scripts
│   ├── run_experiments.py # Main experiment runner
│   ├── train.py           # Single training run
│   └── evaluate.py        # Single evaluation run
├── training/              # Training utilities
│   ├── pipeline.py        # Training/evaluation pipeline
│   ├── trainer.py         # Trainer class
│   ├── losses.py          # Loss functions (ACT loss)
│   └── evaluation.py      # Evaluation functions
├── utils/                 # Utility functions
│   ├── config.py          # Configuration loading
│   └── logging.py         # Logging utilities
├── runs/                  # Output directory for results
│   └── [experiment_name]/ # Results organized by experiment
├── commands.txt           # Pre-built experiment commands
└── requirements.txt       # Python dependencies
```

---

## Configuration Files

### Main Configuration (`configs/main.yaml`)

This is the primary configuration file containing all experiment settings.

#### Key Sections to Modify:

```yaml
# ============================================
# SECTION 1: Global Settings
# ============================================
seed: 42                    # Random seed for reproducibility
device: cuda                # Use 'cuda' for GPU, 'cpu' for CPU

# ============================================
# SECTION 2: Experiment Settings (CHANGE THESE)
# ============================================
experiment:
  name: gm_hrm_full         # ⚠️ CHANGE: Name your experiment
  output_root: runs         # ⚠️ CHANGE: Where to save results
  save_checkpoint: true     # Save model checkpoints
  checkpoint_name: model.pt # Checkpoint filename

# ============================================
# SECTION 3: Data Settings
# ============================================
data:
  benchmark: sudoku         # Current benchmark (auto-set by runner)
  root: data/sudoku-mini    # ⚠️ CHANGE: Path to your training data
  train_split: train        # Training data folder name
  test_split: test          # Test data folder name
  max_train_samples: null   # Limit training samples (null = all)
  max_test_samples: null    # Limit test samples (null = all)
  normalize_divisor: 10.0   # Input normalization factor
  target_index: 0           # Which output to predict
  num_workers: 0            # DataLoader workers
  pin_memory: true          # CUDA memory optimization
  auto_generate: false      # ⚠️ CHANGE: true to auto-generate datasets

# ============================================
# SECTION 4: Model Architecture
# ============================================
model:
  name: gm_hrm              # Model type (only 'gm_hrm' supported)
  hidden_dim: 256           # Hidden dimension size
  input_dim: 81             # Input dimension (changes per benchmark)
  output_dim: 10            # Output classes (changes per benchmark)
  use_gating: true          # Enable gated fusion

# ============================================
# SECTION 5: Memory Module
# ============================================
memory:
  enabled: true             # Enable external memory
  slots: 16                 # Number of memory slots
  slot_dim: 256             # Memory slot dimension
  reset_each_forward: true  # Reset memory between batches

# ============================================
# SECTION 6: Halting/ACT Settings
# ============================================
halting:
  enabled: true             # Enable adaptive computation
  adaptive: true            # Use learned halting
  max_segments: 10          # Maximum reasoning steps
  ponder_cost: 0.01         # Pondering penalty
  threshold: 1.0            # Halting threshold

# ============================================
# SECTION 7: Training Settings
# ============================================
training:
  batch_size: 64            # ⚠️ CHANGE: Reduce if OOM errors
  lr: 3.0e-4                # Learning rate
  epochs: 50                # ⚠️ CHANGE: Number of training epochs
  weight_decay: 1.0e-2      # L2 regularization
  grad_clip_norm: 1.0       # Gradient clipping
  max_batches_per_epoch: null  # Limit batches (null = all)

# ============================================
# SECTION 8: Optimizer Settings
# ============================================
optimizer:
  name: adamw               # Optimizer type
  betas: [0.9, 0.95]        # Adam betas
  eps: 1.0e-8               # Adam epsilon

# ============================================
# SECTION 9: Evaluation Settings
# ============================================
evaluation:
  max_batches: null         # Limit eval batches (null = all)

# ============================================
# SECTION 10: Benchmarks to Run
# ============================================
benchmarks:
  enabled: [sudoku, maze, math, arc]  # ⚠️ CHANGE: Select benchmarks
  definitions:
    # ... (benchmark-specific settings)
```

### Smoke Test Configuration (`configs/smoke.yaml`)

A minimal configuration for quick testing:

```yaml
extends: main.yaml          # Inherits from main.yaml

experiment:
  name: gm_hrm_smoke

data:
  auto_generate: true       # Auto-generate small datasets
  max_train_samples: 64
  max_test_samples: 32

training:
  epochs: 1
  batch_size: 16
  max_batches_per_epoch: 2

evaluation:
  max_batches: 2
```

---

## Running Experiments

### Option 1: Run All Experiments (Full Suite)

This runs all benchmarks with all ablation variants:

```bash
python scripts/run_experiments.py --config configs/main.yaml
```

### Option 2: Run Smoke Test (Quick Validation)

Test that everything works with a quick run:

```bash
python scripts/run_experiments.py --config configs/smoke.yaml
```

### Option 3: Run Specific Benchmark-Variant Combinations

Use the pre-built commands in `commands.txt`:

```bash
# Example: Run Sudoku with HRM baseline only
python -c "import yaml, pathlib; p=pathlib.Path('configs/main.yaml'); cfg=yaml.safe_load(p.read_text()); cfg['benchmarks']['enabled']=['sudoku']; cfg['ablations']['enabled']=True; cfg['ablations']['variants']=[v for v in cfg['ablations']['variants'] if v['name']=='hrm_baseline']; pathlib.Path('configs/auto_sudoku_hrm_baseline.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))" && python scripts/run_experiments.py --config configs/auto_sudoku_hrm_baseline.yaml
```

### Option 4: Single Training Run

```bash
python scripts/train.py --config configs/main.yaml --run-dir runs/my_experiment
```

### Option 5: Single Evaluation Run

```bash
python scripts/evaluate.py --config configs/main.yaml --run-dir runs/my_experiment
```

---

## What to Change Before Running

### ⚠️ CRITICAL: Settings You MUST Review

| Setting | Location | What to Change | Example |
|---------|----------|----------------|---------|
| **Output Root** | `experiment.output_root` | Where results are saved | `runs`, `./outputs`, `D:/results` |
| **Experiment Name** | `experiment.name` | Unique name for this run | `exp_v1`, `ablation_study_01` |
| **Device** | `device` | GPU or CPU | `cuda`, `cpu` |
| **Batch Size** | `training.batch_size` | Reduce if out of memory | `64`, `32`, `16` |
| **Epochs** | `training.epochs` | Training duration | `50`, `100`, `200` |
| **Benchmarks** | `benchmarks.enabled` | Which benchmarks to run | `[sudoku]`, `[maze, math]` |
| **Auto Generate** | `data.auto_generate` | Auto-generate datasets | `true`, `false` |

### Dataset Paths

Each benchmark has its own dataset path. Modify these in `benchmarks.definitions`:

```yaml
benchmarks:
  definitions:
    sudoku:
      levels:
        - name: bt22
          dataset:
            root: data/benchmarks/sudoku_bt22  # ⚠️ CHANGE this path
```

### Memory Considerations

If running on limited GPU memory:

```yaml
training:
  batch_size: 16        # Reduce from 64
  
memory:
  slots: 8              # Reduce from 16
  slot_dim: 128         # Reduce from 256
```

---

## Output and Results

### Directory Structure

After running experiments, results are organized as:

```
runs/
├── {experiment_name}/
│   ├── {benchmark}/
│   │   ├── {level}/
│   │   │   ├── {variant}/
│   │   │   │   └── model.pt          # Model checkpoint
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── {experiment_name}_summary.json     # All results in JSON
├── {experiment_name}_summary.csv      # All results in CSV
└── {experiment_name}_plots/           # Generated plots
    ├── {benchmark}_eval_accuracy.png
    ├── {benchmark}_eval_loss.png
    └── {benchmark}_train_last_loss.png
```

### Summary JSON Format

```json
[
  {
    "benchmark": "sudoku",
    "level": "bt22",
    "variant": "gm_hrm_full",
    "run_dir": "runs/gm_hrm_full/sudoku/bt22/gm_hrm_full",
    "device": "cuda",
    "steps": 7820,
    "train_last_loss": 0.142857,
    "eval_loss": 0.185432,
    "eval_accuracy": 0.9523,
    "eval_samples": 1000
  },
  ...
]
```

### Generated Plots

Three plots per benchmark:
- `eval_accuracy.png` - Test accuracy comparison
- `eval_loss.png` - Test loss comparison
- `train_last_loss.png` - Final training loss comparison

---

## Individual Scripts

### `scripts/run_experiments.py`

Main experiment runner that:
1. Loads configuration
2. Iterates through enabled benchmarks
3. For each benchmark, iterates through difficulty levels
4. For each level, runs all ablation variants
5. Saves results to JSON, CSV, and generates plots

**Arguments:**
```bash
python scripts/run_experiments.py --config <path_to_config>
```

### `scripts/train.py`

Single training run:

```bash
python scripts/train.py \
    --config configs/main.yaml \
    --run-dir runs/my_single_run
```

**Arguments:**
- `--config`: Path to YAML config (default: `configs/main.yaml`)
- `--run-dir`: Output directory (default: from config)

### `scripts/evaluate.py`

Single evaluation run:

```bash
python scripts/evaluate.py \
    --config configs/main.yaml \
    --run-dir runs/my_single_run \
    --checkpoint runs/my_single_run/model.pt
```

**Arguments:**
- `--config`: Path to YAML config
- `--run-dir`: Directory containing checkpoint
- `--checkpoint`: Optional explicit checkpoint path

---

## Benchmarks and Ablations

### Available Benchmarks

| Benchmark | Description | Input Dim | Levels |
|-----------|-------------|-----------|--------|
| **Sudoku** | Sudoku puzzle solving | 81 (9×9) | bt22, bt50, bt100, bt200 |
| **Maze** | Path finding in mazes | 900-10000 | g30, g50, g75, g100 |
| **Math** | Arithmetic chain reasoning | 64 | chain_5_7, chain_8_11, chain_12_15 |
| **ARC** | ARC-AGI tasks | 900 | arc_agi_1, arc_agi_2 |

### Ablation Variants

| Variant | Gating | Memory | ACT | Description |
|---------|--------|--------|-----|-------------|
| `hrm_baseline` | ❌ | ❌ | Adaptive | Basic HRM without enhancements |
| `gm_hrm_full` | ✅ | ✅ | Adaptive | Full GM-HRM model |
| `hrm_plus_memory_no_gate` | ❌ | ✅ | Adaptive | Memory without gating |
| `hrm_plus_gate_no_memory` | ✅ | ❌ | Adaptive | Gating without memory |
| `gm_hrm_fixed_act` | ✅ | ✅ | Fixed | Fixed computation steps |
| `gm_hrm_large_memory` | ✅ | ✅ (32 slots) | Adaptive | Larger memory capacity |

### Running Specific Configurations

**Single benchmark, single variant:**
```python
# In configs/main.yaml, change:
benchmarks:
  enabled: [sudoku]  # Only sudoku

ablations:
  enabled: true
  variants:
    - name: gm_hrm_full
      overrides: {}
    # Comment out other variants
```

**All benchmarks, single variant:**
```python
# In configs/main.yaml:
benchmarks:
  enabled: [sudoku, maze, math, arc]

ablations:
  enabled: true
  variants:
    - name: gm_hrm_full
      overrides: {}
```

---

## Quick Reference Commands

### Full Experiment Suite
```bash
# All benchmarks, all variants (takes several hours)
python scripts/run_experiments.py --config configs/main.yaml
```

### Smoke Test
```bash
# Quick validation (~5 minutes)
python scripts/run_experiments.py --config configs/smoke.yaml
```

### Per-Benchmark Commands

**Sudoku Only:**
```bash
python -c "import yaml, pathlib; p=pathlib.Path('configs/main.yaml'); cfg=yaml.safe_load(p.read_text()); cfg['benchmarks']['enabled']=['sudoku']; pathlib.Path('configs/sudoku_only.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))" && python scripts/run_experiments.py --config configs/sudoku_only.yaml
```

**Maze Only:**
```bash
python -c "import yaml, pathlib; p=pathlib.Path('configs/main.yaml'); cfg=yaml.safe_load(p.read_text()); cfg['benchmarks']['enabled']=['maze']; pathlib.Path('configs/maze_only.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))" && python scripts/run_experiments.py --config configs/maze_only.yaml
```

**Math Only:**
```bash
python -c "import yaml, pathlib; p=pathlib.Path('configs/main.yaml'); cfg=yaml.safe_load(p.read_text()); cfg['benchmarks']['enabled']=['math']; pathlib.Path('configs/math_only.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))" && python scripts/run_experiments.py --config configs/math_only.yaml
```

**ARC Only:**
```bash
python -c "import yaml, pathlib; p=pathlib.Path('configs/main.yaml'); cfg=yaml.safe_load(p.read_text()); cfg['benchmarks']['enabled']=['arc']; pathlib.Path('configs/arc_only.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))" && python scripts/run_experiments.py --config configs/arc_only.yaml
```

### Dataset Generation

If datasets are missing, generate them:

```bash
# Sudoku (downloads from HuggingFace)
python -m data.sudoku_generator --output-dir data/benchmarks/sudoku_bt22 --min-difficulty 22

# Maze (downloads from HuggingFace)
python -m data.maze_generator --source-repo sapientinc/maze-30x30-hard-1k --output-dir data/benchmarks/maze_30

# Math chains (generates locally)
python -m data.math_generator --output-dir data/benchmarks/math_5_7 --min-steps 5 --max-steps 7

# Mock data (for testing)
python -m data.mock_generator --output-dir data/benchmarks/arc_agi_1 --seq-len 900 --vocab-size 12 --output-dim 12
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size in config:
```yaml
training:
  batch_size: 16  # Try 32, 16, 8
```

#### 2. Dataset Not Found
```
FileNotFoundError: Dataset missing for 'sudoku/bt22'
```
**Solution:** Either:
1. Set `data.auto_generate: true` in config
2. Manually generate: `python -m data.sudoku_generator --output-dir data/benchmarks/sudoku_bt22`

#### 3. CUDA Not Available
```
UserWarning: CUDA not available, falling back to CPU
```
**Solution:** Either:
1. Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
2. Or set `device: cpu` in config (slower)

#### 4. Import Errors
```
ModuleNotFoundError: No module named 'data.common'
```
**Solution:** Run from repository root:
```bash
cd /path/to/Heirarchical_Reasoning_Model
python scripts/run_experiments.py --config configs/main.yaml
```

#### 5. HuggingFace Download Issues
```
requests.exceptions.ConnectionError
```
**Solution:** Set HuggingFace token or use offline mode:
```bash
export HF_TOKEN=your_token
# or generate data locally with mock_generator
```

### Checking Results

```bash
# View summary JSON
cat runs/gm_hrm_full_summary.json | python -m json.tool

# View CSV in terminal
column -s, -t < runs/gm_hrm_full_summary.csv | head -20

# List all checkpoints
find runs -name "*.pt" -type f
```

---

## Customization Guide

### Adding a New Ablation Variant

1. Add to `ablations.variants` in `configs/main.yaml`:

```yaml
ablations:
  variants:
    - name: my_custom_variant
      overrides:
        model:
          use_gating: true
        memory:
          enabled: true
          slots: 32
        halting:
          max_segments: 20
```

### Adding a New Benchmark

1. Create generator in `data/`:
```python
# data/my_generator.py
def main():
    # Generate train/test splits
    # Save as all__inputs.npy, all__labels.npy
```

2. Add to `configs/main.yaml`:
```yaml
benchmarks:
  enabled: [sudoku, maze, math, arc, my_benchmark]
  definitions:
    my_benchmark:
      levels:
        - name: level1
          dataset:
            root: data/benchmarks/my_benchmark_l1
            input_dim: 100
            output_dim: 10
          generator:
            module: data.my_generator
            args:
              - --output-dir
              - data/benchmarks/my_benchmark_l1
```

### Changing the Model

Edit `models/gm_hrm.py` or create a new model class and update `training/pipeline.py`:

```python
def build_model(cfg):
    model_name = str(cfg.model.name).lower()
    if model_name == "my_model":
        return MyModel(cfg)
    elif model_name == "gm_hrm":
        return GMHRM(cfg)
```

---

## Complete Example: Running Full Ablation Study

```bash
# Step 1: Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run smoke test first
python scripts/run_experiments.py --config configs/smoke.yaml

# Step 4: Check smoke test results
cat runs/gm_hrm_smoke_summary.json

# Step 5: Run full experiments (modify config first!)
# Edit configs/main.yaml:
#   - Set experiment.name to your experiment name
#   - Set experiment.output_root to your output folder
#   - Adjust training.epochs as needed
#   - Select benchmarks in benchmarks.enabled

# Step 6: Run full suite
python scripts/run_experiments.py --config configs/main.yaml

# Step 7: Check results
ls runs/
cat runs/gm_hrm_full_summary.json
```

---

## Contact & Citation

For questions about this codebase, please refer to the paper or contact the authors.

If you use this code, please cite:
```bibtex
@article{gmhrm2024,
  title={Gated-Memory Hierarchical Reasoning Model},
  author={...},
  year={2024}
}
```
