WANDB_PROJECT="malaysian-sft" \
WANDB_NAME="fsdp2-fused-moe-Qwen3-30B-A3B-lora-256" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 --master_port=29502 \
-m lora-qwen3-moe \
--model_name "gfs/01be5b33/Qwen3-30B-A3B-Instruct-2507-stack" \
--batch_size 4 \
--grad_accumulation 1
