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

## Run pretrained inference

Use these steps when you want to run inference using pretrained ViT weights (either from `timm` or a local checkpoint).

1. Quick inference using a timm model name (no local checkpoint file required):

```bash
# Runs evaluation using a timm pretrained backbone (replace dataset path if performing full validation)
python src/eval.py experiment=validate_vit \
  ckpt_path=vit_base_patch16_224 \
  trainer.devices=1 \
  trainer.accelerator=gpu \
  data.data_dir=/path/to/ILSVRC2012
```

Notes:
- `ckpt_path` can be a timm model identifier (e.g. `vit_base_patch16_224`) — the code will call `timm.create_model(..., pretrained=True)` and load weights automatically.
- If you only want to run inference on a few images, you can supply a small local folder structured like ImageNet `val/<class>/*.jpg` and point `data.data_dir` to it.

2. Inference using a local checkpoint file:

```bash
python src/eval.py experiment=validate_vit \
  ckpt_path=/full/path/to/checkpoint.ckpt \
  trainer.devices=1 \
  trainer.accelerator=gpu \
  data.data_dir=/path/to/ILSVRC2012
```

3. Single-image quick test (no full ImageNet required):
- Option A: Use the visualization script to inspect patch selection (no model needed):

```bash
python scripts/gen_visualization_single.py --input /path/to/image.jpg --output /tmp/vis.jpg --method entropy --vis_type grid

python gen_visualization_single.py --input /Users/home_folder/Desktop/Python_projects/VAQ-apt/scripts/test_image.png --output /tmp/vis.jpg --method entropy --vis_type entropy

```

- Option B: Quick inference script you can save as `scripts/run_single_inference.py` and run (this loads a timm pretrained model and runs a forward pass on one image):

```python
from PIL import Image
import torch
from torchvision import transforms
from src.models.vision_transformer import VisionTransformer

# Replace model name and img size as needed
model_name = 'vit_base_patch16_224'
img_size = 224

# Build model instance consistent with config (adjust parameters if needed)
net = VisionTransformer(img_size=img_size, patch_size=16, num_classes=1000)

# Load pretrained via timm through ViTLitModule-style logic (example):
import timm
timm_model = timm.create_model(model_name, pretrained=True, img_size=img_size)
state = timm_model.state_dict()
net.load_state_dict(state, strict=False)

# Prepare image
img = Image.open('/path/to/image.jpg').convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
inp = preprocess(img).unsqueeze(0)

net.eval()
with torch.no_grad():
    logits = net.forward_features(inp)
    # If needed, run through head
    print(logits.shape)

```

4. Where pretrained weights live
- timm downloads pretrained backbones into the timm/torch cache (e.g. `~/.cache/torch` or `~/.cache/timm`). Local checkpoints you pass via `ckpt_path` can live anywhere; if saved by Lightning they will typically appear under the Hydra output folder `logs/<task_name>/runs/<timestamp>/checkpoints/` unless you override `callbacks.model_checkpoint.dirpath`.

5. Important caveats
- `src/eval.py` asserts that `cfg.ckpt_path` is set — when using a timm model name set `ckpt_path` to that name.
- Full ImageNet validation requires ImageNet data; for quick functional checks, use a small custom folder or the single-image script above.

