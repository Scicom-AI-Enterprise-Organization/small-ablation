WANDB_PROJECT="malaysian-sft" \
WANDB_NAME="fsdp1-gpt-oss-120b-lora-256" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 --master_port=29502 \
-m lora-gpt-oss \
--fsdp "full_shard auto_wrap" \
--fsdp_config fsdp-gpt-oss.json \
--model_name_or_path ramdisk/gpt-oss-120b-BF16 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--output_dir fsdp1-gpt-oss-120b-lora-256 \
--bf16 --do_train --do_eval false --num_train_epochs 3 \
--train_file "multipacking-gpt-oss" \
--logging_steps 1 \
--learning_rate 1e-4 \
--warmup_steps 50 \
--weight_decay 0.01 \
--save_strategy "epoch" \
--save_total_limit 10 \
--model_dtype bfloat16 \
--rank 256 \
--alpha 512 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 4 \
--include_expert_lora true \
--save_only_model true \
--include_num_input_tokens_seen true