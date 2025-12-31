# Malaysian SFT

LoRA SFT on https://huggingface.co/datasets/Scicom-intl/Malaysian-Instructions

## Ablation on multiple models

1. Ablation on Qwen/Qwen3-32B, Qwen/Qwen2.5-72B-Instruct, meta-llama/Llama-3.1-70B-Instruct, openai/gpt-oss-120b, zai-org/GLM-4.5-Air and Qwen/Qwen3-235B-A22B
2. Dense LoRA SFT done using DeepSpeed Zero3 HF Trainer while MoE LoRA SFT done using FSDP2 + Fused MoE except for GPT OSS.
3. Multipacking variable length 16384 context length, with global batch size of 32, so global total tokens is 524288.
4. All self attention linear layers with rank 256 with alpha multiply by 2.0 <sup> + </sup>
5. Liger fused cross entropy.
6. 1e-4 learning rate, 50 warmup, 3 epoch only.
7. Calculate accuracy for each epoch using reasoning system prompt.

<sup> + </sup> with the rank of each equal to the total rank divided by the number of active experts, https://thinkingmachines.ai/blog/lora/

## WanDB

https://wandb.ai/aies-scicom-scicom-ai/malaysian-sft

## Benchmark

We benchmark using https://huggingface.co/datasets/UMxYTLAILabs/MalayMMLU

### Qwen/Qwen3-32B

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

malaymmlu-ds3-qwen3-32b-lora-256-checkpoint-960
STEM 0.8088415882112158 0.8935734752353663
Language 0.7865776081424937 0.8956743002544529
Social science 0.7108991037872218 0.8519803411390575
Others 0.7505396977692492 0.8671144159270808
Humanities 0.7742889647326507 0.8873720136518771
average 0.7662293925285664 0.8791429092415669

malaymmlu-ds3-qwen3-32b-lora-256-checkpoint-640
STEM 0.8112975849365535 0.9029881293491608
Language 0.773854961832061 0.8891539440203562
Social science 0.7055507372072853 0.8531367447239087
Others 0.7399856080594867 0.8743103861837371
Humanities 0.764505119453925 0.8869169510807736
average 0.7590388022978622 0.8813012310715873

malaymmlu-ds3-qwen3-32b-lora-256-checkpoint-320
STEM 0.8076135898485469 0.904625460499386
Language 0.7652671755725191 0.8901081424936387
Social science 0.6971668112171148 0.8425845620121423
Others 0.7378268169824898 0.8649556248500839
Humanities 0.7458475540386803 0.8773606370875996
average 0.7507443895318702 0.87592688538857
```

### Qwen/Qwen2.5-72B-Instruct

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

malaymmlu-ds3-qwen2.5-72b-lora-256-checkpoint-321
STEM 0.8080229226361032 0.9029881293491608
Language 0.750795165394402 0.8866094147582697
Social science 0.7108991037872218 0.8528476438276958
Others 0.7236747421443991 0.8702326697049653
Humanities 0.7199089874857793 0.8764505119453925
average 0.742660184289581 0.8778256739170969

malaymmlu-ds3-qwen2.5-72b-lora-256-checkpoint-642
STEM 0.7572656569791241 0.8763814981580025
Language 0.7544529262086515 0.8829516539440203
Social science 0.6711477305579647 0.8314541775079503
Others 0.6828975773566802 0.847445430558887
Humanities 0.6621160409556314 0.8166097838452787
average 0.7055759864116105 0.8509685088028277

malaymmlu-ds3-qwen2.5-72b-lora-256-checkpoint-963
STEM 0.7765042979942693 0.8898894801473598
Language 0.7643129770992366 0.8885178117048346
Social science 0.7013587742122 0.8560277536860365
Others 0.7183976972895179 0.8683137443031902
Humanities 0.7051194539249147 0.8516496018202503
average 0.7331386401040277 0.8708796783323344
```

### meta-llama/Llama-3.1-70B-Instruct

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

malaymmlu-ds3-llama3.1-70b-lora-256-checkpoint-323
STEM 0.7932869422840769 0.896029471960704
Language 0.7655852417302799 0.8856552162849872
Social science 0.721740387395201 0.8508239375542064
Others 0.7248740705205086 0.863036699448309
Humanities 0.7802047781569966 0.8998862343572241
average 0.7571382840174125 0.8790863119210861

malaymmlu-ds3-llama3.1-70b-lora-256-checkpoint-969
STEM 0.7961522717969709 0.9029881293491608
Language 0.7794211195928753 0.8894720101781171
Social science 0.7237640936686903 0.8509684880023128
Others 0.7323099064523867 0.8611177740465339
Humanities 0.7908987485779294 0.8951080773606371
average 0.7645092280177705 0.8799308957873524

malaymmlu-ds3-llama3.1-70b-lora-256-checkpoint-646
STEM 0.7932869422840769 0.8902988129349161
Language 0.7698791348600509 0.8861323155216285
Social science 0.7302688638334779 0.8527030933795895
Others 0.7366274886063804 0.8606380426960902
Humanities 0.7899886234357224 0.8923777019340159
average 0.7640102106039417 0.8764299932932479
```

### zai-org/GLM-4.5-Air

### Qwen/Qwen3-30B-A3B-Instruct-2507