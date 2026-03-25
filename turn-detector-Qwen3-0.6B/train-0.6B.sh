#!/bin/bash
# Train 0.6B model on H100

WANDB_API_KEY= \
WANDB_PROJECT=qwen3-0.6B-turn-detector-v2 \
HF_TOKEN= \
CUDA_VISIBLE_DEVICES="0" \
python train.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --model_dtype bfloat16 \
  --flash_attn_version flash_attention_2 \
  --train_file ./parquet-train-merged-v2 \
  --test_file ./parquet-test-v2 \
  --block_size 8192 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --lr_scheduler_type constant \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 50 \
  --save_total_limit 10 \
  --output_dir ./output-0.6B-v2 \
  --bf16 true \
  --gradient_checkpointing true \
  --include_num_input_tokens_seen true \
  --report_to wandb \
  --run_name qwen3-0.6B-turn-detector-2 \
  --dataloader_num_workers 0 \
  --eval_strategy steps \
  --eval_steps 250
