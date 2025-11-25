import json
import os
import re
from typing import Optional, Union

import bitsandbytes
import safetensors.torch
import torch
from huggingface_hub.serialization import save_torch_state_dict
from peft import LoraConfig
from tqdm import tqdm
from transformers import Qwen3MoeConfig


TStateDict = dict[str, torch.Tensor]


def load_sharded_state_dict(save_directory: os.PathLike) -> TStateDict:
    index_path = os.path.join(save_directory, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        model_path = os.path.join(save_directory, "model.safetensors")
        return safetensors.torch.load_file(model_path)

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    state_dict = {}
    for shard_file in tqdm(shard_files):
        # Load shard into memory
        shard_path = os.path.join(save_directory, shard_file)
        state_dict |= safetensors.torch.load_file(shard_path)
    return state_dict


def find_key_prefix(state_dict: TStateDict, *, is_fused: bool) -> str:
    if is_fused:
        key_base = "layers.0.mlp.down_proj.weight"
        key_base_lora = "layers.0.mlp.down_proj.lora_A.weight"
    else:
        key_base = "layers.0.mlp.experts.0.down_proj.weight"
        key_base_lora = "layers.0.mlp.experts.0.down_proj.lora_A.weight"

    for key_prefix in [
        "",
        "model.",
        "model.model.",
        "base_model.",
        "base_model.model.",
        "base_model.model.model.",
    ]:
        if key_prefix + key_base in state_dict or key_prefix + key_base_lora in state_dict:
            return key_prefix
    raise RuntimeError("Key prefix not found.")


def convert_state_dict_to_fused_(
    state_dict: TStateDict, *, param_names: list[str], num_hidden_layers: int, num_experts: int
) -> None:
    key_prefix = find_key_prefix(state_dict, is_fused=False)
    for layer_idx in range(num_hidden_layers):
        print(f"Layer {layer_idx}/{num_hidden_layers}")
        for param_name in param_names:
            params = []
            for expert_idx in tqdm(range(num_experts)):
                key = f"{key_prefix}layers.{layer_idx}.mlp.experts.{expert_idx}.{param_name}.weight"
                params.append(state_dict[key])
                del state_dict[key]
            key = f"{key_prefix}layers.{layer_idx}.mlp.{param_name}.weight"
            state_dict[key] = torch.stack(params)
            del params


def convert_state_dict_to_unfused_(
    state_dict: TStateDict, *, param_names: list[str], num_hidden_layers: int, num_experts: int
) -> None:
    key_prefix = find_key_prefix(state_dict, is_fused=True)
    for layer_idx in range(num_hidden_layers):
        print(f"Layer {layer_idx}/{num_hidden_layers}")
        for param_name in param_names:
            params_key = f"{key_prefix}layers.{layer_idx}.mlp.{param_name}.weight"
            params = state_dict[params_key]
            for expert_idx in tqdm(range(num_experts)):
                key = f"{key_prefix}layers.{layer_idx}.mlp.experts.{expert_idx}.{param_name}.weight"
                # A clone is needed, otherwise the state dict cannot be saved
                state_dict[key] = params[expert_idx].clone()
            del state_dict[params_key]
            del params


def convert_model_to_fused(
    in_dir: os.PathLike, out_dir: os.PathLike, *, max_shard_size: Optional[Union[int, str]] = None
) -> None:
    print(f"Loading {in_dir}")
    config = Qwen3MoeConfig.from_pretrained(in_dir)
    config.architectures[0] = config.architectures[0].replace("Qwen3Moe", "Qwen3MoeFused")
    state_dict = load_sharded_state_dict(in_dir)

    print("Converting...")
    convert_state_dict_to_fused_(
        state_dict,
        param_names=[
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        num_hidden_layers=config.num_hidden_layers,
        num_experts=config.num_experts,
    )

    print(f"Saving {out_dir}")
    config.save_pretrained(out_dir)
    max_shard_size_kwarg = {} if max_shard_size is None else {"max_shard_size": max_shard_size}
    save_torch_state_dict(state_dict, out_dir, **max_shard_size_kwarg)


def convert_model_to_unfused(
    in_dir: os.PathLike, out_dir: os.PathLike, *, max_shard_size: Optional[Union[int, str]] = None
) -> None:
    print(f"Loading {in_dir}")
    config = Qwen3MoeConfig.from_pretrained(in_dir)
    config.architectures[0] = config.architectures[0].replace("Qwen3MoeFused", "Qwen3Moe")
    state_dict = load_sharded_state_dict(in_dir)

    print("Converting...")
    convert_state_dict_to_unfused_(
        state_dict,
        param_names=[
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        num_hidden_layers=config.num_hidden_layers,
        num_experts=config.num_experts,
    )

    print(f"Saving {out_dir}")
    config.save_pretrained(out_dir)
    max_shard_size_kwarg = {} if max_shard_size is None else {"max_shard_size": max_shard_size}
    save_torch_state_dict(state_dict, out_dir, **max_shard_size_kwarg)


def convert_lora_to_fused(
    in_dir: os.PathLike, out_dir: os.PathLike, *, max_shard_size: Optional[Union[int, str]] = None
) -> None:
    print(f"Loading {in_dir}")
    config = LoraConfig.from_pretrained(in_dir)
    model_path = os.path.join(in_dir, "adapter_model.safetensors")
    state_dict = safetensors.torch.load_file(model_path)

    key_prefix = find_key_prefix(state_dict, is_fused=False)
    pattern = key_prefix + r"layers\.\d+\.mlp\.experts\.0\.down_proj\.lora_A\.weight"
    num_hidden_layers = len([x for x in state_dict if re.compile(pattern).fullmatch(x)])
    pattern = key_prefix + r"layers\.0\.mlp\.experts\.\d+\.down_proj\.lora_A\.weight"
    num_experts = len([x for x in state_dict if re.compile(pattern).fullmatch(x)])
    print("num_hidden_layers", num_hidden_layers, "num_experts", num_experts)

    print("Converting...")
    convert_state_dict_to_fused_(
        state_dict,
        param_names=[
            "gate_proj.lora_A",
            "gate_proj.lora_B",
            "up_proj.lora_A",
            "up_proj.lora_B",
            "down_proj.lora_A",
            "down_proj.lora_B",
        ],
        num_hidden_layers=num_hidden_layers,
        num_experts=num_experts,
    )

    print(f"Saving {out_dir}")
    config.save_pretrained(out_dir)
    model_path = os.path.join(out_dir, "adapter_model.safetensors")
    safetensors.torch.save_file(state_dict, model_path)


def convert_lora_to_unfused(
    in_dir: os.PathLike, out_dir: os.PathLike, *, max_shard_size: Optional[Union[int, str]] = None
) -> None:
    print(f"Loading {in_dir}")
    config = LoraConfig.from_pretrained(in_dir)
    model_path = os.path.join(in_dir, "adapter_model.safetensors")
    state_dict = safetensors.torch.load_file(model_path)

    key_prefix = find_key_prefix(state_dict, is_fused=True)
    pattern = key_prefix + r"layers\.\d+\.mlp\.down_proj\.lora_A\.weight"
    num_hidden_layers = len([x for x in state_dict if re.compile(pattern).fullmatch(x)])
    num_experts = state_dict[key_prefix + "layers.0.mlp.down_proj.lora_A.weight"].shape[0]
    print("num_hidden_layers", num_hidden_layers, "num_experts", num_experts)

    print("Converting...")
    convert_state_dict_to_unfused_(
        state_dict,
        param_names=[
            "gate_proj.lora_A",
            "gate_proj.lora_B",
            "up_proj.lora_A",
            "up_proj.lora_B",
            "down_proj.lora_A",
            "down_proj.lora_B",
        ],
        num_hidden_layers=num_hidden_layers,
        num_experts=num_experts,
    )

    print(f"Saving {out_dir}")
    config.save_pretrained(out_dir)
    model_path = os.path.join(out_dir, "adapter_model.safetensors")
    safetensors.torch.save_file(state_dict, model_path)


def convert_optimizer_state_to_fused(
    in_dir: os.PathLike,
    out_dir: os.PathLike,
    *,
    num_hidden_layers: int = 48,
    num_experts: int = 128,
    keys_need_fuse: Optional[list[str]] = None,
    keys_no_need_fuse: Optional[list[str]] = None,
) -> None:
    if keys_need_fuse is None:
        keys_need_fuse = [
            "gate_proj.lora_A",
            "gate_proj.lora_B",
            "up_proj.lora_A",
            "up_proj.lora_B",
            "down_proj.lora_A",
            "down_proj.lora_B",
        ]
    if keys_no_need_fuse is None:
        keys_no_need_fuse = [
            "q_proj.lora_A",
            "q_proj.lora_B",
            "k_proj.lora_A",
            "k_proj.lora_B",
            "v_proj.lora_A",
            "v_proj.lora_B",
            "o_proj.lora_A",
            "o_proj.lora_B",
        ]

    qmap1 = bitsandbytes.functional.create_dynamic_map(signed=True).cuda()
    qmap2 = bitsandbytes.functional.create_dynamic_map(signed=False).cuda()

    print(f"Loading {in_dir}")
    state_path = os.path.join(in_dir, "optimizer.pt")
    # Quantization needs to run on GPU
    state_dict_old = torch.load(state_path, map_location="cuda")
    num_keys_old = (len(keys_no_need_fuse) + len(keys_need_fuse) * num_experts) * num_hidden_layers
    assert len(state_dict_old["state"]) == num_keys_old

    print("Converting...")
    state_dict_new = {
        "state": {},
        "param_groups": [],
    }
    state_dict_new["param_groups"] = state_dict_old["param_groups"]
    num_keys_new = (len(keys_no_need_fuse) + len(keys_need_fuse)) * num_hidden_layers
    state_dict_new["param_groups"][0]["params"] = list(range(num_keys_new))

    key_idx_old = 0
    key_idx_new = 0
    for _ in tqdm(range(num_hidden_layers)):
        for _ in range(len(keys_no_need_fuse)):
            state_dict_new["state"][key_idx_new] = state_dict_old["state"][key_idx_old]
            key_idx_old += 1
            key_idx_new += 1
        for param_idx in range(len(keys_need_fuse)):
            quant_state_new = {}
            already_quantized = "qmap1" in state_dict_old["state"][key_idx_old + param_idx]
            for attr in ["step", "qmap1", "qmap2"] if already_quantized else ["step"]:
                # No need to fuse the attr
                _key_idx_old = key_idx_old + param_idx
                quant_state_new[attr] = state_dict_old["state"][_key_idx_old][attr]

            for attr in ["state1", "state2", "absmax1", "absmax2"] if already_quantized else ["state1", "state2"]:
                # Fuse the attr
                tensors = []
                for expert_idx in range(num_experts):
                    _key_idx_old = key_idx_old + len(keys_need_fuse) * expert_idx + param_idx
                    tensors.append(state_dict_old["state"][_key_idx_old][attr])
                if attr.startswith("state"):
                    tensors = torch.stack(tensors)
                else:
                    tensors = torch.cat(tensors)
                quant_state_new[attr] = tensors

            if not already_quantized and quant_state_new["state1"].numel() >= 4096:
                quant_state_new["qmap1"] = qmap1
                quant_state_new["state1"], _quant_state = bitsandbytes.functional.quantize_blockwise(
                    quant_state_new["state1"], code=qmap1, blocksize=256, nested=False
                )
                quant_state_new["absmax1"] = _quant_state.absmax

                quant_state_new["qmap2"] = qmap2
                quant_state_new["state2"], _quant_state = bitsandbytes.functional.quantize_blockwise(
                    quant_state_new["state2"], code=qmap2, blocksize=256, nested=False
                )
                quant_state_new["absmax2"] = _quant_state.absmax

            state_dict_new["state"][key_idx_new] = quant_state_new
            key_idx_new += 1
        key_idx_old += len(keys_need_fuse) * num_experts
    assert key_idx_old == num_keys_old
    assert key_idx_new == num_keys_new

    print(f"Saving {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, "optimizer.pt")
    torch.save(state_dict_new, state_path)