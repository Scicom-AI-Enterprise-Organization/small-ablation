from transformers import Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from tqdm import tqdm
from torch import nn
import torch

model = Qwen3MoeForCausalLM.from_pretrained('gfs/01be5b33/Qwen3-30B-A3B-Instruct-2507', torch_dtype="auto")

for module in tqdm(model.modules()):
    if isinstance(module, Qwen3MoeSparseMoeBlock):
        gate_proj_stacked = torch.stack([e.gate_proj.weight.data for e in module.experts], dim=0)
        up_proj_stacked = torch.stack([e.up_proj.weight.data for e in module.experts], dim=0)
        down_proj_stacked = torch.stack([e.down_proj.weight.data for e in module.experts], dim=0)

        module.gate_proj = nn.Parameter(gate_proj_stacked)
        module.up_proj = nn.Parameter(up_proj_stacked)
        module.down_proj = nn.Parameter(down_proj_stacked)

        del module.experts

model.save_pretrained('gfs/01be5b33/Qwen3-30B-A3B-Instruct-2507-non-transpose')