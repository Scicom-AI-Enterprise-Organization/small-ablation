#!/usr/bin/env python3
#
# Randomly initialize a tiny model
# Then it can be trained in example_train_tiny.py

from transformers import AutoTokenizer, Qwen3MoeConfig

from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM


def main():
    model_dir = "./pretrained/qwen-moe-tiny-lm"

    # Create the model
    config = Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=5,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_window_layers=2,
        moe_intermediate_size=3,
        num_experts=11,
        norm_topk_prob=True,
    )
    model = Qwen3MoeFusedForCausalLM(config)
    model.save_pretrained(model_dir)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.save_pretrained(model_dir)

    model = Qwen3MoeFusedForCausalLM.from_pretrained(model_dir)
    model.save_pretrained(model_dir)

    tokenizer.save_pretrained(model_dir)


if __name__ == "__main__":
    main()