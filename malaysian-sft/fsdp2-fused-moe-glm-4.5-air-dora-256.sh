WANDB_PROJECT="malaysian-sft" \
WANDB_NAME="fsdp2-fused-moe-glm-4.5-air-dora-256" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 --master_port=29502 \
-m dora-glm-air