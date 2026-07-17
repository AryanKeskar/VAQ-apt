from xml.parsers.expat import model

import transformers
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class Qwen2VLModel(nn.Module):
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        super(Qwen2VLModel, self).__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def forward(self, inputs):
        output_ids = self.model.generate(**inputs, max_new_tokens=50)
        return output_ids