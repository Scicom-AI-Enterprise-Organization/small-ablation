WANDB_PROJECT="fsdp-vs-hsdp-vs-fsdp-tp" \
WANDB_NAME="fsdp-tp" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 \
-m fsdp-tp