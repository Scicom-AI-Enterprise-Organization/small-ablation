WANDB_PROJECT="fsdp-vs-hsdp-vs-fsdp-tp" \
WANDB_NAME="fsdp" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 \
-m fsdp