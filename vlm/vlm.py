import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import requests

# 1. Setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
model_id = "Qwen/Qwen2-VL-2B-Instruct"

print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,   # FIX 1: float16 is much more stable on MPS
    device_map="auto",
    attn_implementation="eager"  # FIX 2: Bypasses the low-level MPS attention crash
)
processor = AutoProcessor.from_pretrained(model_id)

# 2. Grab a real image
image_path = "/Users/home_folder/Desktop/Python_projects/VAQ-apt/vlm/test_image.png"
image = Image.open(image_path)

# 3. Format the prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image}, 
            {"type": "text", "text": "Describe this image in one sentence."}
        ]
    }
]

# 4. Process inputs
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=image, padding=True, return_tensors="pt").to(device)
keys = inputs.keys()
input_ids = list(inputs.input_ids)
with open("inputs_structure.py", "w") as f:
    f.write("inputs_metadata = {\n")
    f.write(f"{inputs}\n")
    f.write("}\n")

print("Saved structure to inputs_structure.py")
# 5. Generate Inference
print("Running inference...")
output_ids = model.generate(**inputs, max_new_tokens=50)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

print("\n--- INFERENCE OUTPUT ---")
print(output_text)
print("------------------------\n")

# 6. Dump the architecture
print("--- FULL MODEL ARCHITECTURE ---")
print(model)

"""

--- FULL MODEL ARCHITECTURE ---
Qwen2VLForConditionalGeneration(
  (visual): Qwen2VisionTransformerPretrainedModel(
    (patch_embed): PatchEmbed(
      (proj): Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
    )
    (rotary_pos_emb): VisionRotaryEmbedding()
    (blocks): ModuleList(
      (0-31): 32 x Qwen2VLVisionBlock(
        (norm1): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
        (norm2): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
        (attn): VisionAttention(
          (qkv): Linear(in_features=1280, out_features=3840, bias=True)
          (proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (mlp): VisionMlp(
          (fc1): Linear(in_features=1280, out_features=5120, bias=True)
          (act): QuickGELUActivation()
          (fc2): Linear(in_features=5120, out_features=1280, bias=True)
        )
      )
    )
    (merger): PatchMerger(
      (ln_q): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
      (mlp): Sequential(
        (0): Linear(in_features=5120, out_features=5120, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=5120, out_features=1536, bias=True)
      )
    )
  )
  (model): Qwen2VLModel(
    (embed_tokens): Embedding(151936, 1536)
    (layers): ModuleList(
      (0-27): 28 x Qwen2VLDecoderLayer(
        (self_attn): Qwen2VLAttention(
          (q_proj): Linear(in_features=1536, out_features=1536, bias=True)
          (k_proj): Linear(in_features=1536, out_features=256, bias=True)
          (v_proj): Linear(in_features=1536, out_features=256, bias=True)
          (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
          (rotary_emb): Qwen2VLRotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (up_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (down_proj): Linear(in_features=8960, out_features=1536, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((1536,), eps=1e-06)
    (rotary_emb): Qwen2VLRotaryEmbedding()
  )
  (lm_head): Linear(in_features=1536, out_features=151936, bias=False)
)
"""