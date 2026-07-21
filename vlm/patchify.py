import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
#Import the entropy_utils module
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import Qwen2VLModel
from transformers import AutoProcessor
from src.models.entropy_utils import *
from src.models.patch_tokenizer import PatchTokenizer
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
image_path = "/Users/home_folder/Desktop/Python_projects/VAQ-apt/vlm/test_image.png"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMAGE_SIZE = 336
BASE_PATCH_SIZE = 14
NUM_SCALES = 3
THRESHOLDS = [6.0, 6.0]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 1) Load image and make it RGB
image_path = "/Users/home_folder/Desktop/Python_projects/VAQ-apt/vlm/test_image.png"
image = Image.open(image_path).convert("RGB")
print(f"Original image size: {image.size}")
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
    if isinstance(inputs[key], torch.Tensor):
        #print(f"Shape of the keys: {key}: {list(inputs[key].shape)}")
        pass
    else:
        #print(f"Shape of the keys: {key}: {inputs[key]}")
        pass
resized_patches_14 = inputs['resized_patches_14']
resized_patches_28 = inputs['resized_patches_28']
resized_patches_56 = inputs['resized_patches_56']
print(f"\n\nshape of original patches:----------->")
print(f'shape of resized_patches_14: {resized_patches_14.shape}')
print(f'shape of resized_patches_28: {resized_patches_28.shape}')
print(f'shape of resized_patches_56: {resized_patches_56.shape}')
print(f"-----------------------------\n")
# 1. Stack all the resized patches into one sequence
# 1. Stack all the resized patches into one sequence
all_patches = torch.cat([
    inputs['resized_patches_14'],  
    inputs['resized_patches_28'],  
    inputs['resized_patches_56']   
], dim=0) 

pixel_values = all_patches.unsqueeze(1).repeat(1, 2, 1, 1, 1).view(all_patches.size(0), -1)

# --- FIX: Match the Processor's Expected 24x24 Grid ---

# A 336x336 image has 24x24 patches. We must pad our 405 patches 
# up to 576 so the ViT outputs exactly 144 features.
expected_side = 24
target_patches = expected_side * expected_side  # 576

num_patches = pixel_values.size(0)
pad_len = target_patches - num_patches

# Pad the sequence with zeros 
if pad_len > 0:
    padding = torch.zeros((pad_len, pixel_values.size(1)), dtype=pixel_values.dtype, device=pixel_values.device)
    pixel_values = torch.cat([pixel_values, padding], dim=0)

# Set the grid to exactly what the text processor expects: [T=1, H=24, W=24]
image_grid_thw = torch.tensor([[1, expected_side, expected_side]])

# ------------------------------------------------------

# Inputting this into the model
model_id = "Qwen/Qwen2-VL-2B-Instruct"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in one sentence."}
        ]
    }
]

processor = AutoProcessor.from_pretrained(model_id)
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=image, padding=True, return_tensors="pt").to(device)

# Overwrite with our custom, perfectly-padded vision tensors
inputs['pixel_values'] = pixel_values.to(device)
inputs['image_grid_thw'] = image_grid_thw.to(device)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,   
    device_map="auto",
    attn_implementation="eager"  
)

output_ids = model.generate(**inputs, max_new_tokens=50)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

print(f"model's output text: {output_text}")