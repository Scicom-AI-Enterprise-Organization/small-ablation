WANDB_PROJECT="malaysian-sft" \
WANDB_NAME="fsdp2-fused-moe-Qwen3-235B-A22B-lora-256" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 --master_port=29502 \
-m lora-qwen3-moe \
--model_name "nfs/Qwen3-235B-A22B-Instruct-2507-stack" \
--batch_size 1 \
--grad_accumulation 4
