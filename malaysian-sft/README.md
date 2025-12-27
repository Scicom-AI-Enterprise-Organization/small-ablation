# Malaysian SFT

LoRA SFT on https://huggingface.co/datasets/Scicom-intl/Malaysian-Instructions

## Ablation on multiple models

1. Ablation on Qwen/Qwen3-32B, Qwen/Qwen2.5-72B-Instruct, meta-llama/Llama-3.1-70B-Instruct, openai/gpt-oss-120b, zai-org/GLM-4.5-Air and Qwen/Qwen3-235B-A22B
2. Dense LoRA SFT done using DeepSpeed Zero3 HF Trainer while MoE LoRA SFT done using FSDP2 + EP except for GPT OSS.
3. Multipacking variable length 16384 context length, with global batch size of 32, so global total tokens is 524288.
4. All self attention linear layers with rank 256 with alpha multiply by 2.0 <sup> + </sup>
5. Liger fused cross entropy.
6. 1e-4 learning rate, 50 warmup, 3 epoch only.
7. Calculate accuracy for each epoch using non-reasoning and reasoning system prompt.

<sup> + </sup> with the rank of each equal to the total rank divided by the number of active experts, https://thinkingmachines.ai/blog/lora/

## WanDB

https://wandb.ai/aies-scicom-scicom-ai/malaysian-sft

## Benchmark

We benchmark using https://huggingface.co/datasets/UMxYTLAILabs/MalayMMLU

### Reasoning

#### Qwen/Qwen3-32B

1. Run generation,

```bash
python3 malaymmlu.py --pattern "ds3-qwen3-32b-lora-256*" --num_gpus 8 --gpu_partition 2
```

2. Calculate accuracy,

```bash
python3 calculate_malaymmlu.py --pattern "malaymmlu-ds3-qwen3-32b-lora-256-checkpoint-*"
```

```
malaymmlu-ds3-qwen3-32b-lora-256-checkpoint-960 72634
malaymmlu-ds3-qwen3-32b-lora-256-checkpoint-640 72629
malaymmlu-ds3-qwen3-32b-lora-256-checkpoint-320 72629
malaymmlu-ds3-qwen3-32b-lora-256-checkpoint-960 0.7588913214093932 0.8766987483993556
malaymmlu-ds3-qwen3-32b-lora-256-checkpoint-640 0.7507745693394473 0.8775147684554054
malaymmlu-ds3-qwen3-32b-lora-256-checkpoint-320 0.7422927514670634 0.8718902388627159
```

#### Qwen/Qwen2.5-72B-Instruct

1. Run generation,

```bash
python3 malaymmlu.py --pattern "ds3-qwen2.5-72b-lora-256*" --num_gpus 8 --gpu_partition 4
```

2. Calculate accuracy,

```bash
python3 calculate_malaymmlu.py --pattern "malaymmlu-ds3-qwen2.5-72b-lora-256-checkpoint-*"
```

```
malaymmlu-ds3-qwen2.5-72b-lora-256-checkpoint-321 72637
malaymmlu-ds3-qwen2.5-72b-lora-256-checkpoint-642 72638
malaymmlu-ds3-qwen2.5-72b-lora-256-checkpoint-963 72639
malaymmlu-ds3-qwen2.5-72b-lora-256-checkpoint-321 0.7362932925063103 0.8756155087516034
malaymmlu-ds3-qwen2.5-72b-lora-256-checkpoint-642 0.7099172863230011 0.859177876180132
malaymmlu-ds3-qwen2.5-72b-lora-256-checkpoint-963 0.7331145634294259 0.8742211514496968
```

#### meta-llama/Llama-3.1-70B-Instruct

1. Run generation,

```bash
python3 malaymmlu.py --pattern "ds3-llama3.1-70b-lora-256*" --num_gpus 8 --gpu_partition 4
```

2. Calculate accuracy,

```bash
python3 calculate_malaymmlu.py --pattern "malaymmlu-ds3-llama3.1-70b-lora-256-checkpoint-*"
```

```
malaymmlu-ds3-llama3.1-70b-lora-256-checkpoint-323 72616
malaymmlu-ds3-llama3.1-70b-lora-256-checkpoint-969 72638
malaymmlu-ds3-llama3.1-70b-lora-256-checkpoint-646 72638
malaymmlu-ds3-llama3.1-70b-lora-256-checkpoint-323 0.7516212978644305 0.8755834606964352
malaymmlu-ds3-llama3.1-70b-lora-256-checkpoint-969 0.7591789534547557 0.8759757155247181
malaymmlu-ds3-llama3.1-70b-lora-256-checkpoint-646 0.7588485524305125 0.8737455086110767
```