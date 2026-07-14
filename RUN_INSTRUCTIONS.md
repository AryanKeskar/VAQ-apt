# Running the VAQ-APT codebase

This document summarizes the minimal, actionable steps to get the VAQ-APT repository running in your `apt` Conda environment. It collects environment setup, dataset configuration, common CLI commands, and troubleshooting tips.

## Quick checklist
- Conda environment: activated (`conda activate apt`).
- Python: 3.10 recommended.
- PyTorch: installed with correct CUDA/MPS support for your machine.
- Python deps: installed (`pip install -r requirements.txt`).
- Data path: set in `configs/data/imagenet.yaml` (see below).

## Environment setup
1. (Optional) Create the conda environment:

```bash
conda create -n apt python=3.10 -y
conda activate apt
```

2. Install PyTorch for your hardware (adjust CUDA version or choose `mps` on macOS):

Example (CUDA 12.1):
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

3. Install project Python dependencies (you already ran this):

```bash
pip install -r requirements.txt
```

## Project-specific configuration (edit these files)
- `configs/data/imagenet.yaml`: set the `data_dir` to the folder that contains `train/` and `val/`.

Example content to paste into `configs/data/imagenet.yaml`:

```yaml
data_dir: /path/to/ILSVRC2012
# other keys may stay as-is
```

- If you see module import errors using `rootutils`, create an empty file at repository root named `.project-root`.

## How to run
Run all commands from the repository root.

1) Validate a model (example overrides):

```bash
python src/eval.py experiment=validate_vit \
  trainer.devices=4 \
  data.batch_size=128 \
  ckpt_path=/path/to/checkpoint.ckpt \
  data.data_dir=/path/to/ILSVRC2012
```

Notes:
- Replace `trainer.devices`, `trainer.accelerator` (`gpu`/`mps`) with values appropriate to your machine.
- You can alternatively edit the config file under `configs/trainer/`.

2) Train / fine-tune (example single-GPU):

```bash
python src/train.py experiment=train_vit_finetune \
  trainer.devices=1 \
  trainer.accelerator=gpu \
  data.data_dir=/path/to/ILSVRC2012 \
  data.batch_size=64
```

3) Generate a visualization:

```bash
python scripts/gen_visualization_single.py \
  --input /your/input/image.jpg \
  --output /your/output/visualization.jpg \
  --method entropy \
  --vis_type grid
```

## Hydra notes
- Hydra changes working directory by default (creates `outputs/` entries). Running from repo root is normal.
- If you want more detailed error traces, run with:

```bash
HYDRA_FULL_ERROR=1 python src/train.py experiment=train_vit_finetune ...
```

## Common troubleshooting
- rootutils / import errors: create `./.project-root` in repo root and run commands from repo root.
- Dataset not found: double-check `data.data_dir` path and that `train/` and `val/` exist.
- CUDA driver mismatch: reinstall PyTorch with the CUDA version matching your system.
- Multi-GPU errors: try single-device runs first, or ensure `strategy` in trainer config matches Lightning version; you may need `torchrun` for some distributed setups.

## Files to inspect
- `README.md` — project overview and recommended install hints.
- `src/train.py` and `src/eval.py` — main entry points for training and evaluation.
- `configs/data/imagenet.yaml` — dataset path and dataset-specific settings.
- `configs/trainer/*.yaml` — trainer settings for CPU/GPU/DDP.
- `requirements.txt` — Python dependencies.

---

If you want, I can also produce a ready-to-paste `configs/data/imagenet.yaml` example or a minimal single-image quick test config; tell me which.
