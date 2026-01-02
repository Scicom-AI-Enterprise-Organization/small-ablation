from transformers import Glm4MoeForCausalLM
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMoE
from tqdm import tqdm
from torch import nn
import torch

model = Glm4MoeForCausalLM.from_pretrained('ramdisk/GLM-4.5-Air', torch_dtype="auto")

for module in tqdm(model.modules()):
    if isinstance(module, Glm4MoeMoE):
        gate_proj_stacked = torch.stack([e.gate_proj.weight.data for e in module.experts], dim=0).transpose(1, 2)
        up_proj_stacked = torch.stack([e.up_proj.weight.data for e in module.experts], dim=0).transpose(1, 2)
        down_proj_stacked = torch.stack([e.down_proj.weight.data for e in module.experts], dim=0).transpose(1, 2)

        module.gate_proj = nn.Parameter(gate_proj_stacked)
        module.up_proj = nn.Parameter(up_proj_stacked)
        module.down_proj = nn.Parameter(down_proj_stacked)

        del module.experts

model.save_pretrained('ramdisk/GLM-4.5-Air-stack')