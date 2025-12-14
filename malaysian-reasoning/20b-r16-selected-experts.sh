CUDA_VISIBLE_DEVICES="6,7" \
WANDB_PROJECT="malaysian-reasoning-20b" \
WANDB_NAME="lora-r16-selected-experts" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 2 --master_port=29501 \
-m openai-oss-sft-lora \
--fsdp "full_shard auto_wrap" \
--fsdp_config fsdp.json \
--model_name_or_path openai/gpt-oss-20b \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2 \
--output_dir malaysian-reasoning-20b-lora-r16-selected-experts \
--bf16 --do_train --do_eval false --num_train_epochs 2 \
--train_file "malaysian-reasoning-16k-mosaic" \
--logging_steps 1 \
--learning_rate 2e-4 \
--warmup_steps 50 \
--weight_decay 0.01 \
--save_steps 10000 \
--save_total_limit 3 \
--model_dtype bfloat16 \
--rank 16 \
--alpha 32 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 4 \
--include_expert_lora true \
--save_only_model true \
--specific_layers "3,2,18,1"