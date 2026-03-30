WANDB_API_KEY= \
WANDB_PROJECT=qwen3-1.7B-turn-detector-v2 \
WANDB_NAME=qwen3-1.7B-turn-detector-2\
CUDA_VISIBLE_DEVICES="0" \
python train.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --model_dtype bfloat16 \
  --flash_attn_version "kernels-community/flash-attn3" \
  --train_file ./train/train \
  --block_size 8192 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --lr_scheduler_type constant \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 50 \
  --save_total_limit 10 \
  --output_dir ./output-1.7B \
  --bf16 true \
  --include_num_input_tokens_seen true \
  --report_to wandb \
  --dataloader_num_workers 4