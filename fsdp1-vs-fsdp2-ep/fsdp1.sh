WANDB_PROJECT="fsdp1-vs-fsdp2-ep" \
WANDB_NAME="fsdp1" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 --master_port=29503 \
-m fsdp1 \
--fsdp "full_shard auto_wrap" \
--fsdp_config fsdp.json \
--model_name_or_path openai/gpt-oss-20b \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 1 \
--output_dir fsdp1-dummy \
--bf16 --do_train --do_eval false --max_steps 200 \
--train_file "malaysian-reasoning-16k-mosaic" \
--logging_steps 1 \
--learning_rate 1e-4 \
--warmup_steps 50 \
--weight_decay 0.01 \
--save_steps 10000 \
--save_total_limit 3 \
--model_dtype bfloat16 \
--rank 256 \
--alpha 512 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 4 \
--include_expert_lora true \
--save_only_model true \
--include_num_input_tokens_seen true