import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
#Import the entropy_utils module
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.entropy_utils import *
from src.models.patch_tokenizer import PatchTokenizer
image_path = "/Users/home_folder/Desktop/Python_projects/VAQ-apt/vlm/test_image.png"

IMAGE_SIZE = 336
BASE_PATCH_SIZE = 14
NUM_SCALES = 3
THRESHOLDS = [6.0, 6.0]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 1) Load image and make it RGB
image_path = "/Users/home_folder/Desktop/Python_projects/VAQ-apt/vlm/test_image.png"
image = Image.open(image_path).convert("RGB")
image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

# 2) Convert to tensor in [0, 255]
image_pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()

# 3) Compute importance maps for entropy selection
importance_maps = compute_patch_entropy_batched(
    image_pixels,
    patch_size=BASE_PATCH_SIZE,
    num_scales=NUM_SCALES,
)

# 4) Normalize for PatchTokenizer input
mean = torch.tensor(MEAN).view(1, 3, 1, 1)
std = torch.tensor(STD).view(1, 3, 1, 1)
image_tensor = (image_pixels / 255.0 - mean) / std

# 5) Create the tokenizer
patch_tokenizer = PatchTokenizer(
    num_scales=NUM_SCALES,
    base_patch_size=BASE_PATCH_SIZE,
    image_size=[IMAGE_SIZE, IMAGE_SIZE],
    thresholds=THRESHOLDS,
    mean=MEAN,
    std=STD,
    method="entropy",
)

# 6) Tokenize/patchify
inputs = patch_tokenizer(image_tensor, importance_maps=importance_maps)

with open("apt_inputs_structure.py", "w") as f:
    f.write("inputs_metadata = {\n")
    f.write(f"{inputs}\n")
    f.write("}\n")

for key in inputs.keys():
    print(f"Shape of the keys: {key}: {list(inputs[key].shape)}")
