WANDB_DISABLED="true" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 --master_port=29503 \
-m openai-oss-sft-lora \
--fsdp "full_shard auto_wrap" \
--fsdp_config fsdp.json \
--model_name_or_path openai/gpt-oss-120b \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2 \
--output_dir malaysian-reasoning-120b-lora-r64-experts \
--bf16 --do_train --do_eval false --max_steps 100 \
--train_file "malaysian-reasoning-16k-mosaic" \
--logging_steps 1 \
--learning_rate 2e-4 \
--weight_decay 0.01 \
--save_steps 10000 \
--save_total_limit 3 \
--lr_scheduler_type "constant" \
--model_dtype bfloat16 \
--rank 64 \
--alpha 128 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 4 \
--include_expert_lora true