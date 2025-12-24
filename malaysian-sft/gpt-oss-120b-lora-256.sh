WANDB_PROJECT="malaysian-sft" \
WANDB_NAME="fsdp2-ep-gpt-oss-120b-lora-256" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 --master_port=29502 \
-m lora-gpt-oss
