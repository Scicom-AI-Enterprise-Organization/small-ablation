WANDB_PROJECT="fsdp1-vs-fsdp2-ep" \
WANDB_NAME="fsdp2-ep" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 --master_port=29502 \
-m fsdp2-ep
