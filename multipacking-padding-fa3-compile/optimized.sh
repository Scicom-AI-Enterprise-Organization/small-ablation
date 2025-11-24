export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WANDB_PROJECT="multipacking-padding-fa3-compile" \
WANDB_NAME="multipacking-optimized" \
TORCH_DISTRIBUTED_DEBUG="info" \
TORCH_LOGS=recompiles \
CUDA_VISIBLE_DEVICES="0" \
python3 train.py \
--model_name_or_path "Qwen/Qwen3-0.6B-Base" \
--per_device_train_batch_size 32 \
--gradient_accumulation_steps 1 \
--bf16 \
--train_file "multipacking-optimized" \
--logging_steps 1 \
--learning_rate 2e-5 \
--warmup_steps 100 \
--max_steps 300 \
--block_size 10240 \
--gradient_checkpointing true \
--dataloader_num_workers 10 \
--dataloader_prefetch_factor 10 \
--remove_unused_columns false \
--include_num_input_tokens_seen true \
--lr_scheduler_type "constant_with_warmup" \
--model_dtype "bfloat16"