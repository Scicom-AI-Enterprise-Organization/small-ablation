WANDB_PROJECT="fsdp2-fused-moe-lora-bf16-vs-fp8" \
WANDB_NAME="bf16" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 \
-m bf16