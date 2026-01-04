from modeling_minimax_m2 import MiniMaxM2SparseMoeBlock, MiniMaxM2ForCausalLM
from tqdm import tqdm
from torch import nn
import torch

model = AutoModelForCausalLM.from_pretrained('ramdisk/MiniMax-M2-Instruct', torch_dtype="auto")

for module in tqdm(model.modules()):
    if isinstance(module, MiniMaxM2SparseMoeBlock):
        
        w1_stacked = torch.stack([e.w1.weight.data for e in module.experts], dim=0).transpose(1, 2)
        w2_stacked = torch.stack([e.w2.weight.data for e in module.experts], dim=0).transpose(1, 2)
        w3_stacked = torch.stack([e.w3.weight.data for e in module.experts], dim=0).transpose(1, 2)

        module.w1 = nn.Parameter(w1_stacked)
        module.w2 = nn.Parameter(w2_stacked)
        module.w3 = nn.Parameter(w3_stacked)

        del module.experts

model.save_pretrained('ramdisk/MiniMax-M2-Instruct-stack')