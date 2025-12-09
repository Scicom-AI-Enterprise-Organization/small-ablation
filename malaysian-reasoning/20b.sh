HF_HOME=/root/ramdisk \
WANDB_PROJECT="malaysian-reasoning-20b-lora-128" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 4 \
-m openai-oss-sft-lora \
--deepspeed ds_config_zero3.json \
--model_name_or_path openai/gpt-oss-20b \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2 \
--output_dir malaysian-reasoning-20b \
--bf16 --do_train --do_eval false --num_train_epochs 3 \
--train_file "tokenized-16k/tokenized-0" \
--logging_steps 1 \
--learning_rate 2e-4 \
--warmup_steps 50 \
--weight_decay 0.01 \
--save_steps 50 \
--save_total_limit 3 \
--gradient_checkpointing true \
--model_dtype bfloat16 \
--rank 128 \
--alpha 256 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 4