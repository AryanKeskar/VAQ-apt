# VAQ-apt: How to Train the ViT and Reuse the Weights

This project is a Hydra-based PyTorch Lightning training setup. The main training entry point is:

- `src/train.py`
- Configs live under `configs/`
- The ViT training experiment is defined in `configs/experiment/train_vit_finetune.yaml`

The project expects a GPU-enabled environment for normal training. It also saves model checkpoints automatically under the Hydra output directory.

---

## 1) Prerequisites

You need:

- Python 3.10
- A CUDA-capable GPU (recommended)
- A local copy of the ImageNet dataset
- Terminal access in the project root

This repo is designed to be run from the project root folder, which is the folder containing `src/`, `configs/`, and `README.md`.

---

## 2) Open the project folder

From a terminal:

```bash
cd /Users/home_folder/Desktop/Python_projects/VAQ-apt
```

Set the project root environment variable that the config files expect:

```bash
export PROJECT_ROOT="$PWD"
```

You should do this in every new terminal session before running training.

---

## 3) Create the Python environment

Recommended setup using conda:

```bash
conda create -n vaqapt python=3.10 -y
conda activate vaqapt
```

Then install PyTorch and the repo dependencies. For example:

```bash
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If you prefer `mamba`, the project README also suggests this workflow:

```bash
mamba create -n apt python=3.10 -y
mamba activate apt
mamba install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

The key dependency stack is already listed in `requirements.txt`, including:

- `lightning`
- `hydra-core`
- `torchmetrics`
- `timm`
- `transformers`
- `rootutils`
- `opencv-python`

---

## 4) Prepare the ImageNet dataset

The repo expects ImageNet-style folders named `train` and `val`.

Your dataset should look like this:

```text
/path/to/ILSVRC2012/
  train/
    n01440764/
    n01443537/
    ...
  val/
    n01440764/
    n01443537/
    ...
```

The exact config file is `configs/data/imagenet.yaml`, and the field that controls the dataset location is `data.data_dir` inside that file:

```yaml
# file: configs/data/imagenet.yaml
_data_dir: ""  # TODO: add! Path to ImageNet dataset
```

In this project, the actual path is supplied at runtime via Hydra overrides like `data.data_dir=/path/to/ILSVRC2012`, because the default value is intentionally blank.

A typical setup is:

```bash
mkdir -p /path/to/
# make sure ImageNet is already downloaded at /path/to/ILSVRC2012
```

Then point the training command at that folder with a CLI override:

```bash
data.data_dir=/path/to/ILSVRC2012
```

---

## 5) Train the ViT

The default training experiment is:

```bash
python src/train.py experiment=train_vit_finetune
```

This is the project’s main ViT training recipe. It uses the config in `configs/experiment/train_vit_finetune.yaml`, which sets:

- ImageNet data
- ViT model config
- APT patch logic
- trainer settings
- a checkpoint callback

You must also explicitly pass your dataset path, because the default config leaves `data_dir` blank.

### Minimal working command

```bash
python src/train.py \
  experiment=train_vit_finetune \
  data=imagenet \
  data.data_dir=/path/to/ILSVRC2012
```

### Example with a single GPU

```bash
python src/train.py \
  experiment=train_vit_finetune \
  data=imagenet \
  data.data_dir=/path/to/ILSVRC2012 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=5
```

### Example with multiple GPUs

```bash
python src/train.py \
  experiment=train_vit_finetune \
  data=imagenet \
  data.data_dir=/path/to/ILSVRC2012 \
  trainer.accelerator=gpu \
  trainer.devices=8 \
  trainer.precision=16-mixed
```

The config already sets `trainer.devices: 8` in the experiment file, so if your machine has 8 GPUs you can keep the defaults and only pass the dataset path.

---

## 6) Where the model weights are saved

The project automatically saves checkpoints through Lightning’s model checkpoint callback.

The checkpoint directory is set in `configs/callbacks/default.yaml`:

```yaml
model_checkpoint:
  dirpath: ${paths.output_dir}/checkpoints
  filename: "epoch_{epoch:03d}"
  save_last: True
```

That means the checkpoint files are stored under the Hydra run directory, usually under a path like:

```text
logs/train/runs/YYYY-MM-DD_HH-MM-SS/checkpoints/
```

You will typically see files such as:

```text
logs/train/runs/2026-08-04_10-12-34/checkpoints/epoch_000.ckpt
logs/train/runs/2026-08-04_10-12-34/checkpoints/last.ckpt
```

This is the important part for reuse: the weights are saved automatically during training and the last checkpoint is also kept as `last.ckpt`.

---

## 7) Re-use a saved model

Once you have a checkpoint, you can re-load it for evaluation or resume training.

### Evaluate a checkpoint

```bash
python src/eval.py \
  ckpt_path=/path/to/checkpoints/last.ckpt \
  data=imagenet \
  data.data_dir=/path/to/ILSVRC2012
```

### Resume training from a checkpoint

```bash
python src/train.py \
  experiment=train_vit_finetune \
  data=imagenet \
  data.data_dir=/path/to/ILSVRC2012 \
  ckpt_path=/path/to/checkpoints/last.ckpt
```

This works because `src/train.py` passes `cfg.get("ckpt_path")` to `trainer.fit(...)`.

---

## 8) Important notes about the repo setup

- The project uses Hydra config overrides, so most values are changed with command-line flags.
- `PROJECT_ROOT` must be set before launching runs.
- `data_dir` must be set to your actual ImageNet folder.
- The default training config assumes a multi-GPU environment, but it can be reduced to a smaller GPU or CPU setup with CLI overrides.
- Training outputs and checkpoints are stored under the Hydra log directory, not in a separate hardcoded folder in the repo.

---

## 9) Recommended exact flow for your use case

If your goal is to train the project’s ViT and keep the weights for later reuse, use this sequence:

```bash
cd /Users/home_folder/Desktop/Python_projects/VAQ-apt
export PROJECT_ROOT="$PWD"

conda create -n vaqapt python=3.10 -y
conda activate vaqapt
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python src/train.py \
  experiment=train_vit_finetune \
  data=imagenet \
  data.data_dir=/path/to/ILSVRC2012
```

After training finishes, your checkpoints will be saved in the Hydra output directory, and you can re-use them with:

```bash
python src/eval.py \
  ckpt_path=/path/to/logs/train/runs/.../checkpoints/last.ckpt \
  data=imagenet \
  data.data_dir=/path/to/ILSVRC2012
```

or resume training from the same checkpoint:

```bash
python src/train.py \
  experiment=train_vit_finetune \
  data=imagenet \
  data.data_dir=/path/to/ILSVRC2012 \
  ckpt_path=/path/to/logs/train/runs/.../checkpoints/last.ckpt
```

---

## 10) Quick troubleshooting

### Error: Hydra complains about no dataset path

Set the dataset path explicitly:

```bash
python src/train.py experiment=train_vit_finetune data.data_dir=/path/to/ILSVRC2012
```

### Error: `PROJECT_ROOT` is not set

Run:

```bash
export PROJECT_ROOT="$PWD"
```

### Error: no GPU detected

Run a single-device CPU fallback only if necessary:

```bash
python src/train.py \
  experiment=train_vit_finetune \
  data=imagenet \
  data.data_dir=/path/to/ILSVRC2012 \
  trainer.accelerator=cpu \
  trainer.devices=1
```

### You want to use your own checkpoint directory

You can override the checkpoint directory at runtime, for example:

```bash
python src/train.py \
  experiment=train_vit_finetune \
  data=imagenet \
  data.data_dir=/path/to/ILSVRC2012 \
  callbacks.model_checkpoint.dirpath=/absolute/path/to/my_checkpoints
```

---

This is the project’s intended flow for training a ViT and keeping the resulting weights for later reuse.
