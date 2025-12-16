CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" \
WANDB_PROJECT="malaysian-reasoning-120b" \
WANDB_NAME="lora-r256" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 6 --master_port=29502 \
-m openai-oss-sft-lora \
--fsdp "full_shard auto_wrap" \
--fsdp_config fsdp.json \
--model_name_or_path openai/gpt-oss-120b \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2 \
--output_dir malaysian-reasoning-120b-lora-r256 \
--bf16 --do_train --do_eval false --num_train_epochs 3 \
--train_file "tokenized-gptoss/tokenized-0" \
--logging_steps 1 \
--learning_rate 2e-4 \
--warmup_steps 50 \
--weight_decay 0.01 \
--save_steps 1000 \
--save_total_limit 3 \
--model_dtype bfloat16 \
--rank 256 \
--alpha 512 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 4