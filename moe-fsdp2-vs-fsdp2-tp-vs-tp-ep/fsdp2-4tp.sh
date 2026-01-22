WANDB_PROJECT="moe-fsdp2-vs-fsdp2-tp-vs-tp-ep" \
WANDB_NAME="fsdp2-tp" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 --master_port=29502 \
-m fsdp2-tp \
--tp_size 4