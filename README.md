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

## Cost

Even though H100 SXM is faster but H100 SXM is $2.038/h while RTX 5090 only $0.555/h,

<img src="price.png" width="50%">