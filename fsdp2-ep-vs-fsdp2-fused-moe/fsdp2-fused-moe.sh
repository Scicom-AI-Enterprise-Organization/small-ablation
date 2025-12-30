WANDB_PROJECT="fsdp2-ep-vs-fsdp2-fused-moe" \
WANDB_NAME="fsdp2-fused-moe" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 \
-m fsdp2-fused-moe