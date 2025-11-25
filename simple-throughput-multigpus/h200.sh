export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WANDB_PROJECT="h100-h200-l40s" \
WANDB_NAME="h200" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 4 \
-m train \
--model_name_or_path "Qwen/Qwen3-0.6B-Base" \
--per_device_train_batch_size 32 \
--gradient_accumulation_steps 1 \
--bf16 \
--train_file "multipacking" \
--logging_steps 1 \
--learning_rate 2e-5 \
--warmup_steps 100 \
--max_steps 150 \
--block_size 4096 \
--gradient_checkpointing true \
--dataloader_num_workers 10 \
--dataloader_prefetch_factor 10 \
--remove_unused_columns false \
--include_num_input_tokens_seen true \
--lr_scheduler_type "constant_with_warmup" \
--model_dtype "bfloat16" \
--attn_implementation "flash_attention_3"