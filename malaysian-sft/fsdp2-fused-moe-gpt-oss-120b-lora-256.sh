WANDB_PROJECT="malaysian-sft" \
WANDB_NAME="fsdp2-fused-moe-gpt-oss-120b-lora-256" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 --master_port=29502 \
-m lora-gpt-oss-fsdp2-fused-moe \
--model_name "gfs/01be5b33/gpt-oss-120b-BF16" \
--batch_size 2 \
--grad_accumulation 2