WANDB_PROJECT="malaysian-reasoning-120b-lora-128" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 \
-m openai-oss-sft-lora \
--fsdp "full_shard auto_wrap" \
--fsdp_config fsdp.json \
--model_name_or_path unsloth/gpt-oss-120b-BF16 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2 \
--output_dir malaysian-reasoning-120b \
--bf16 --do_train --do_eval false --num_train_epochs 5 \
--train_file "malaysian-reasoning-16k-mosaic" \
--logging_steps 1 \
--learning_rate 2e-5 \
--warmup_steps 50 \
--weight_decay 0.01 \
--save_steps 50 \
--save_total_limit 3 \
--model_dtype bfloat16 \
--rank 16 \
--alpha 32 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 4