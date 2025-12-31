# Malaysian Reasoning

LoRA SFT on https://huggingface.co/datasets/mesolitica/Malaysian-Reasoning

## Ablation on GPT OSS 20B

1. Use `kernels-community/vllm-flash-attn3` for Flash Attention 3 with Sink.
2. Multipacking variable length 16384 context length, with global batch size of 8, so global total tokens is 65536.
3. All self attention linear layers with rank 16, 32, 64, 128, 256, 512 with alpha multiply by 2.0
4. All expert gate up projection and down projection with rank 16, 32, 64, 128, 256, 512 with alpha multiply by 2.0 <sup> + </sup>
5. Selected expert gate up projection and down projection based on square root mean `exp_avg_sq`, [20b-r64-experts-gradient.sh](20b-r64-experts-gradient.sh), [notebook/sort-optimizer.ipynb](notebook/sort-optimizer.ipynb), top 4 selected layers are 23, 22, 21, 20. <sup> + </sup>
6. Liger fused cross entropy.
7. 2e-4 learning rate, 50 warmup, 2 epoch only.

<sup> + </sup> with the rank of each equal to the total rank divided by the number of active experts, https://thinkingmachines.ai/blog/lora/

### WanDB

https://wandb.ai/aies-scicom-scicom-ai/malaysian-reasoning-20b

<img src="wandb.png" width="50%">

### Benchmark

We benchmark using https://huggingface.co/datasets/UMxYTLAILabs/MalayMMLU

<img src="lora_accuracy.png" width="100%">

1. Merge with base model,

- merge self attention linear layers using PEFT LoRA checkpoints with base model, [notebook/merge-lora-20b.ipynb](notebook/merge-lora-20b.ipynb)
- merge custom made linear and expert layers with base model, [notebook/merge-manual.ipynb](notebook/merge-manual.ipynb)

2. Evaluating merge models using vLLM inside subprocess inside multiprocessing,

```bash
python3 malaymmlu.py --pattern '*-merged' --num_gpus 8
```

3. Calculate accuracy,

```bash
python3 calculate_malaymmlu.py --pattern 'malaymmlu-malaysian-reasoning-20b*'
```

```
malaymmlu-malaysian-reasoning-20b-lora-r512-experts-merged 72639
malaymmlu-malaysian-reasoning-20b-lora-r128-experts-merged 72639
malaymmlu-malaysian-reasoning-20b-lora-r32-selected-experts-merged 72638
malaymmlu-malaysian-reasoning-20b-lora-r128-selected-experts-merged 72639
malaymmlu-malaysian-reasoning-20b-lora-r64-selected-experts-merged 72639
malaymmlu-malaysian-reasoning-20b-lora-r256-selected-experts-merged 72639
malaymmlu-malaysian-reasoning-20b-lora-r256-experts-merged 72639
malaymmlu-malaysian-reasoning-20b-lora-r512-selected-experts-merged 72639
malaymmlu-malaysian-reasoning-20b-lora-r16-experts-merged 72639
malaymmlu-malaysian-reasoning-20b-lora-r16-selected-experts-merged 72638
malaymmlu-malaysian-reasoning-20b-lora-r32-experts-merged 72639
malaymmlu-malaysian-reasoning-20b-lora-r64-experts-merged 72639

malaymmlu-malaysian-reasoning-20b-lora-r512-experts-merged
STEM 0.7118297175603766 0.8862054850593533
Language 0.6612595419847328 0.8646628498727735
Social science 0.6019080659150043 0.8301532234749928
Others 0.6483569201247301 0.8685536099784121
Humanities 0.6532423208191126 0.8687144482366326
average 0.6553193132807913 0.8636579233244328

malaymmlu-malaysian-reasoning-20b-lora-r128-experts-merged
STEM 0.7040523945968072 0.8878428162095784
Language 0.6568066157760815 0.8573473282442748
Social science 0.6013298641225788 0.83506793871061
Others 0.6344447109618614 0.8534420724394339
Humanities 0.6637087599544937 0.8787258248009101
average 0.6520684690823646 0.8624851960809614

malaymmlu-malaysian-reasoning-20b-lora-r32-selected-experts-merged
STEM 0.6852230863692181 0.8796561604584527
Language 0.6339058524173028 0.8486005089058524
Social science 0.5799363978028332 0.8223474992772477
Others 0.6111777404653395 0.8527224754137683
Humanities 0.6147895335608646 0.8614334470989761
average 0.6250065221231116 0.8529520182308594

malaymmlu-malaysian-reasoning-20b-lora-r128-selected-experts-merged
STEM 0.6958657388456816 0.8780188293082276
Language 0.6456743002544529 0.8581424936386769
Social science 0.5881757733448973 0.8243712055507372
Others 0.6315663228591989 0.8560805948668746
Humanities 0.6400455062571103 0.8689419795221843
average 0.6402655283122682 0.8571110205773401

malaymmlu-malaysian-reasoning-20b-lora-r64-selected-experts-merged
STEM 0.697912402783463 0.8923454768726975
Language 0.6434478371501272 0.8575063613231552
Social science 0.5883203237930038 0.825383058687482
Others 0.623650755576877 0.8522427440633246
Humanities 0.6291240045506257 0.8671217292377702
average 0.6364910647708194 0.858919874036886

malaymmlu-malaysian-reasoning-20b-lora-r256-selected-experts-merged
STEM 0.6913630781825624 0.8759721653704462
Language 0.6463104325699746 0.8619592875318066
Social science 0.5806591500433651 0.8216247470367158
Others 0.6253298153034301 0.8462461021827776
Humanities 0.6329920364050057 0.8559726962457338
average 0.6353309025008675 0.852354999673496

malaymmlu-malaysian-reasoning-20b-lora-r256-experts-merged
STEM 0.7114203847728203 0.8878428162095784
Language 0.6563295165394402 0.863708651399491
Social science 0.6046545244290257 0.8292859207863544
Others 0.6560326217318302 0.8663948189014152
Humanities 0.6555176336746302 0.8662116040955632
average 0.6567909362295493 0.8626887622784803

malaymmlu-malaysian-reasoning-20b-lora-r512-selected-experts-merged
STEM 0.6901350798198935 0.8845681539091281
Language 0.6493320610687023 0.8697519083969466
Social science 0.5959814975426424 0.8291413703382481
Others 0.6373230990645239 0.8601583113456465
Humanities 0.6341296928327645 0.872127417519909
average 0.6413802860657054 0.8631494323019757

malaymmlu-malaysian-reasoning-20b-lora-r16-experts-merged
STEM 0.6856324191567744 0.8845681539091281
Language 0.6356552162849872 0.8575063613231552
Social science 0.5884648742411102 0.8295750216825672
Others 0.6270088750299833 0.8522427440633246
Humanities 0.6348122866894198 0.8675767918088737
average 0.634314734280455 0.8582938145574097

malaymmlu-malaysian-reasoning-20b-lora-r16-selected-experts-merged
STEM 0.7015963978714695 0.8935734752353663
Language 0.6240458015267175 0.8532124681933843
Social science 0.5793581960104076 0.8281295172015033
Others 0.6214919644998801 0.857040057567762
Humanities 0.6200227531285551 0.8625711035267349
average 0.629303022607406 0.8589053243449503

malaymmlu-malaysian-reasoning-20b-lora-r32-experts-merged
STEM 0.6946377404830127 0.8902988129349161
Language 0.6445610687022901 0.8557569974554707
Social science 0.6045099739809193 0.8355015900549292
Others 0.6286879347565364 0.8515231470376589
Humanities 0.6359499431171786 0.8577929465301479
average 0.6416693322079874 0.8581746988026246

malaymmlu-malaysian-reasoning-20b-lora-r64-experts-merged
STEM 0.703643061809251 0.8927548096602538
Language 0.6556933842239185 0.8624363867684478
Social science 0.603642671292281 0.834345186470078
Others 0.6368433677140801 0.8541616694650995
Humanities 0.6357224118316268 0.863481228668942
average 0.6471089793742315 0.8614358562065642
```

Where non-experts,

```
malaymmlu-malaysian-reasoning-20b-lora-r16-merged 0.6103171980835949 0.8461093672559061
malaymmlu-malaysian-reasoning-20b-lora-r32-merged 0.6082934082273252 0.8466049892615232
malaymmlu-malaysian-reasoning-20b-lora-r64-merged 0.6182629166150415 0.850989138066328
malaymmlu-malaysian-reasoning-20b-lora-r512-merged 0.6250361376120266 0.8578862594474043
malaymmlu-malaysian-reasoning-20b-lora-r256-merged 0.6211126254491389 0.8535497460042126
```

### What we learnt

1. LoRA weight always in FP32, during merging, merge in FP32, only downcast to base layer precision during add, our custom Linear LoRA module achieved the same loss as PEFT LoRA, https://wandb.ai/aies-scicom-scicom-ai/malaysian-reasoning-20b/runs/bdgdqjhw
2. FSDP 1 is pain, I think everyone should move on from FSDP 1, for `use_orig_params=False` you cannot put trainable and non-trainable weights in the same module, so you have to wrap the trainable weights in separate module such as `nn.ModuleDict`, and if you use `huggingface.Trainer`, make sure you patch `self.accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(self.model)`.
3. expert bias just makes thing harder.

## Scale up to GPT OSS 120B

We use 256 rank parameter for linear layers including experts, model pushed at https://huggingface.co/Scicom-intl/gpt-oss-120b-Malaysian-Reasoning-SFT-v0.1

## Evaluation

1. Run vLLM,

```bash
vllm serve nfs/malaysian-reasoning-120b-lora-r256-experts --max-model-len 10000 --tensor-parallel-size 8 --enable-expert-parallel
```

Achieved,

```
malaymmlu-120b 0.6650559616734811 0.8618923718663528
```

### WanDB

https://wandb.ai/aies-scicom-scicom-ai/malaysian-reasoning-120b