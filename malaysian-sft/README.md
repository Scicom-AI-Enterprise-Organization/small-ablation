# Malaysian Low Rank SFT

Low Rank SFT on https://huggingface.co/datasets/Scicom-intl/Malaysian-Instructions

## Ablation on multiple models

1. Ablation on Qwen/Qwen3-32B, Qwen/Qwen2.5-72B-Instruct, meta-llama/Llama-3.1-70B-Instruct, zai-org/GLM-4.5-Air and Qwen/Qwen3-30B-A3B-Instruct-2507 and Qwen/Qwen3-235B-A22B-Instruct-2507.
2. Dense LoRA SFT done using DeepSpeed Zero3 HF Trainer while MoE LoRA SFT done using FSDP2 + Fused MoE.
3. Also tried DoRA for Qwen/Qwen3-30B-A3B-Instruct-2507.
4. Multipacking variable length 16384 context length, with global batch size of 32, so global total tokens is 524288.
5. All linear layers with rank 256 with alpha multiply by 2.0, for MoE including experts <sup> + </sup>.
6. Liger fused cross entropy.
7. 1e-4 learning rate, 50 warmup steps, and 3 epoch only.
8. Calculate accuracy for each epoch using reasoning system prompt.

<sup> + </sup> with the rank of each equal to the total rank divided by the number of active experts, https://thinkingmachines.ai/blog/lora/

## WanDB

https://wandb.ai/aies-scicom-scicom-ai/malaysian-sft

## Finetuning

We benchmark using https://huggingface.co/datasets/UMxYTLAILabs/MalayMMLU

### Qwen/Qwen3-32B

1. Finetune,

```bash
bash ds3-qwen3-32b-lora-256.sh
```

2. Run generation,

```bash
python3 malaymmlu.py --pattern "ds3-qwen3-32b-lora-256*" --num_gpus 8 --gpu_partition 2
```

3. Calculate accuracy,

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

1. Finetune,

```bash
bash ds3-qwen2.5-72b-lora-256.sh
```

2. Run generation,

```bash
python3 malaymmlu.py --pattern "ds3-qwen2.5-72b-lora-256*" --num_gpus 8 --gpu_partition 4
```

3. Calculate accuracy,

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

1. Finetune,

```bash
bash ds3-llama3.1-70b-lora-256.sh
```

2. Run generation,

```bash
python3 malaymmlu.py --pattern "ds3-llama3.1-70b-lora-256*" --num_gpus 8 --gpu_partition 4
```

3. Calculate accuracy,

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

1. Finetune,

```bash
bash fsdp2-fused-moe-glm-4.5-air-lora-256.sh
```

2. Run generation,

```bash
python3 malaymmlu.py --pattern "GLM-4.5-Air-lora-256-*" --num_gpus 8 --gpu_partition 4
```

3. Calculate accuracy,

```bash
python3 calculate_malaymmlu.py --pattern "malaymmlu-GLM-4.5-Air-lora-256-*"
```

```
malaymmlu-GLM-4.5-Air-lora-256-944 72636
malaymmlu-GLM-4.5-Air-lora-256-629 72633
malaymmlu-GLM-4.5-Air-lora-256-314 72637

malaymmlu-GLM-4.5-Air-lora-256-944
STEM 0.7851002865329513 0.9136307818256242
Language 0.7242366412213741 0.8904262086513995
Social science 0.6905174906042209 0.869615495808037
Others 0.6924922043655553 0.8829455504917246
Humanities 0.7333333333333333 0.9089874857792947
average 0.725135991211487 0.8931211045112158

malaymmlu-GLM-4.5-Air-lora-256-629
STEM 0.7896029471960704 0.90749079001228
Language 0.7247137404580153 0.884382951653944
Social science 0.6952876553917318 0.8577623590633131
Others 0.7016071000239865 0.8774286399616215
Humanities 0.7312855517633675 0.89419795221843
average 0.7284993989666343 0.8842525385819175

malaymmlu-GLM-4.5-Air-lora-256-314
STEM 0.7580843225542366 0.9021694637740483
Language 0.7096055979643766 0.8877226463104325
Social science 0.6673894189071986 0.8639780283318879
Others 0.6852962341088991 0.8851043415687215
Humanities 0.714448236632537 0.8976109215017065
average 0.7069647620334496 0.8873170802973593
```

### Qwen/Qwen3-30B-A3B-Instruct-2507

1. Finetune,

```bash
bash fsdp2-fused-moe-Qwen3-30B-A3B-lora-256.sh
```

2. Run generation,

```bash
python3 malaymmlu.py --pattern "Qwen3-30B-A3B-Instruct-2507-lora-256-*" --num_gpus 8 --gpu_partition 2
```

3. Calculate accuracy,

```bash
python3 calculate_malaymmlu.py --pattern "malaymmlu-Qwen3-30B-A3B-Instruct-2507-lora-256-*"
```

```
malaymmlu-Qwen3-30B-A3B-Instruct-2507-lora-256-319 72632
malaymmlu-Qwen3-30B-A3B-Instruct-2507-lora-256-959 72632
malaymmlu-Qwen3-30B-A3B-Instruct-2507-lora-256-639 72630

malaymmlu-Qwen3-30B-A3B-Instruct-2507-lora-256-319
STEM 0.8125255832992223 0.9029881293491608
Language 0.7536577608142494 0.8853371501272265
Social science 0.6941312518068806 0.8407054061867592
Others 0.731590309426721 0.8548812664907651
Humanities 0.7588168373151308 0.8800910125142207
average 0.7501443485324408 0.8728005929336264

malaymmlu-Qwen3-30B-A3B-Instruct-2507-lora-256-959
STEM 0.7994269340974212 0.9001227998362669
Language 0.7646310432569975 0.8853371501272265
Social science 0.7081526452732003 0.84027175484244
Others 0.7395058767090429 0.8656752218757496
Humanities 0.7592718998862343 0.8873720136518771
average 0.7541976798445793 0.875755788066712

malaymmlu-Qwen3-30B-A3B-Instruct-2507-lora-256-639
STEM 0.8059762586983218 0.898894801473598
Language 0.7631997455470738 0.8867684478371501
Social science 0.7016478751084129 0.8396935530500145
Others 0.7392660110338211 0.8731110578076278
Humanities 0.756769055745165 0.8864618885096701
average 0.7533717892265589 0.8769859497356121
```

### Qwen/Qwen3-30B-A3B-Instruct-2507 DoRA

1. Finetune,

```bash
bash fsdp2-fused-moe-Qwen3-30B-A3B-dora-256.sh
```

2. Run generation,

```bash
python3 malaymmlu.py --pattern "Qwen3-30B-A3B-Instruct-2507-dora-256-*" --num_gpus 8 --gpu_partition 8
```

3. Calculate accuracy,

```bash
python3 calculate_malaymmlu.py --pattern "malaymmlu-Qwen3-30B-A3B-Instruct-2507-dora-256-*"
```

### Qwen/Qwen3-235B-A22B-Instruct-2507

1. Finetune,

```bash
bash fsdp2-fused-moe-Qwen3-235B-A22B-lora-256.sh
```

2. Run generation,

```bash
python3 malaymmlu.py --pattern "Qwen3-235B-A22B-Instruct-2507-lora-256-*" --num_gpus 8 --gpu_partition 8
```

3. Calculate accuracy,

```bash
python3 calculate_malaymmlu.py --pattern "malaymmlu-Qwen3-235B-A22B-Instruct-2507-lora-256-*"
```