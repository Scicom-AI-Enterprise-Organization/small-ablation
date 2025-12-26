WANDB_PROJECT="fsdp-vs-hsdp-vs-fsdp-tp" \
WANDB_NAME="hsdp-2dp" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 \
-m hsdp-2dp