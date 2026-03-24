#!/bin/bash
# Train 1.7B model on H100/H200

CUDA_VISIBLE_DEVICES="0" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py \
    --model_name_or_path Qwen/Qwen3-1.7B-Base \
    --model_dtype bfloat16 \
    --flash_attn_version flash_attention_3 \
    --train_file ./parquet-train-merged \
    --block_size 8192 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_steps 100 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 500 \
    --output_dir ./output-1.7B \
    --bf16 true \
    --gradient_checkpointing true \
    --include_num_input_tokens_seen true \
    --report_to wandb \
    --run_name call-center-ls-1.7B \
    --dataloader_num_workers 4
