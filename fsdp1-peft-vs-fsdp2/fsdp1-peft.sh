WANDB_PROJECT="fsdp1-peft-vs-fsdp2" \
WANDB_NAME="fsdp1-peft" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 --master_port=29502 \
-m fsdp1-peft \
--fsdp "full_shard auto_wrap" \
--fsdp_config fsdp.json \
--model_name_or_path "ramdisk/Qwen3-32B" \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 1 \
--output_dir fsdp1-peft \
--bf16 --do_train --do_eval false --max-steps 100 \
--train_file "multipacking" \
--logging_steps 1 \
--learning_rate 1e-4 \
--warmup_steps 10 \
--weight_decay 0.01 \
--save_strategy "epoch" \
--save_total_limit 10 \
--model_dtype bfloat16 \
--rank 256 \
--alpha 512 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 4 \
--block_size 16384 \
--gradient_checkpointing false \
--save_only_model true \
--include_num_input_tokens_seen true

