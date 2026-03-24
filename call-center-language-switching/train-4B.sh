#!/bin/bash
# Train 4B model on H200/B200
# For B200: change flash_attn_version to flash_attention_4
# For B200: set GPU_PEAK_FLOPS=2250e12

CUDA_VISIBLE_DEVICES="0" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
GPU_PEAK_FLOPS=989e12 \
python train.py \
    --model_name_or_path Qwen/Qwen3-4B-Base \
    --model_dtype bfloat16 \
    --flash_attn_version flash_attention_3 \
    --train_file ./parquet-train-merged \
    --block_size 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_steps 100 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 500 \
    --output_dir ./output-4B \
    --bf16 true \
    --gradient_checkpointing true \
    --include_num_input_tokens_seen true \
    --report_to wandb \
    --run_name call-center-ls-4B \
    --dataloader_num_workers 4
