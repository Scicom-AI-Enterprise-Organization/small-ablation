export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WANDB_PROJECT="liger-vs-cce" \
WANDB_NAME="liger-loss-bf16-bf16-bs4" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_VISIBLE_DEVICES="0" \
python3 train.py \
--model_name_or_path "Qwen/Qwen3-0.6B-Base" \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2 \
--bf16 \
--train_file "multipacking" \
--logging_steps 1 \
--learning_rate 2e-5 \
--warmup_steps 100 \
--max_steps 1000 \
--block_size 10240 \
--gradient_checkpointing false \
--dataloader_num_workers 10 \
--dataloader_prefetch_factor 10 \
--remove_unused_columns false \
--include_num_input_tokens_seen true \
--lr_scheduler_type "constant_with_warmup"