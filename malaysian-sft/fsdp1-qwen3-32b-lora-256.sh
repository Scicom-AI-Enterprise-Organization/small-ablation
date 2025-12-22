CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" \
WANDB_PROJECT="malaysian-sft" \
WANDB_NAME="fsdp1-qwen3-32b-lora-256" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 6 --master_port=29502 \
-m train-lora-dense \
--fsdp "full_shard auto_wrap" \
--fsdp_config fsdp.json \
--model_name_or_path "ramdisk/Qwen3-32B" \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2 \
--output_dir malaysian-reasoning-20b-lora-r512 \
--bf16 --do_train --do_eval false --num_train_epochs 5 \
--train_file "multipacking-qwen3" \
--logging_steps 1 \
--learning_rate 2e-4 \
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
--block_size 16384 \
--gradient_checkpointing false \
--save_only_model true
