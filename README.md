# how-fast-5090

How fast RTX 5090 vs H100 SXM on SLM Qwen3 1.7B full parameter finetuning using PyTorch 2.8.0 CUDA 12.8?

## Hyperparameters

1. 81920 tokens batch size, 1 row is 4096 tokens, so 20 batch size equal to 81920.
2. AdamW optimizer.
3. BF16 weight load same goes to activation, gradient and optimizer states.
4. Gradient checkpointing.
5. https://github.com/apple/ml-cross-entropy Kahan summation FP32 to save up logits activation memory.

## How to

1. Prepare the data, run [prepare-dataset.ipynb](prepare-dataset.ipynb).

Or you can just clone,

```bash
hf download Scicom-intl/mosaic-dummy-fineweb-edu-100b-shuffle --repo-type=dataset --local-dir=./multipacking
```

2. Run the finetuning,

For 5090,

```bash
bash 5090.sh
```

For H100,

```bash
bash h100.sh
```

## WanDB

https://wandb.ai/aies-scicom-scicom-ai/how-fast-5090

<img src="wandb.png" width="50%">

## Cost

Even though H100 SXM is faster but H100 SXM is $2.038/h while RTX 5090 only $0.555/h,

<img src="price.png" width="50%">

Given,

| GPU          | Time                          | Rate ($/hr)   |
| ------------ | ----------------------------- | ------------- |
| **H100 SXM** | 47 min = 47/60 = **0.7833 h** | **2.038 $/h** |
| **RTX 5090** | 96 min = 96/60 = **1.6 h**    | **0.555 $/h** |

### Cost to complete the job

Cost = Rate x Time

H100 SXM:
2.038 x 0.7833 = $1.595

RTX 5090:
0.555 x 1.6 = $0.888

- 5090 is cheaper per job, about 44% lower total cost.

### Performance ratio

Speed ratio = 96/47 = 2.04

- H100 SXM is roughly 2x faster.

### Cost per unit of work (normalized)

| GPU      | Cost per minute ($)  | Relative cost per unit of work |
| -------- | -------------------- | ------------------------------ |
| H100 SXM | 2.038 / 60 = 0.0340  | (1.595 / 47) = 0.0339          |
| RTX 5090 | 0.555 / 60 = 0.00925 | (0.888 / 96) = 0.00925         |

So H100 SXM costs ~3.67x more per minute of compute.

### Conclusion

| Metric                   | H100 SXM     | RTX 5090    | Which is Better                        |
| ------------------------ | ------------ | ----------- | -------------------------------------- |
| Time to complete         | 47 min       | 96 min      | **H100** (~2x faster)                  |
| Cost per run             | $1.60        | $0.89       | **5090** (~44% cheaper)                |
| Cost per unit of compute | 3.67x higher | 1x baseline | **5090**                               |
| Speed per dollar         | 0.56x        | 1x baseline | **5090 (~80% better cost-efficiency)** |
