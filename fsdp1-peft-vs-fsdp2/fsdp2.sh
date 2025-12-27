WANDB_PROJECT="fsdp1-peft-vs-fsdp2" \
WANDB_NAME="fsdp2" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 \
-m fsdp2
