WANDB_PROJECT="how-fast-5090" \
WANDB_NAME="5090" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_VISIBLE_DEVICES="0" \
python3 train.py \
--model_name_or_path "Qwen/Qwen3-1.7B-Base" \
--per_device_train_batch_size 5 \
--gradient_accumulation_steps 4 \
--output_dir Qwen-Qwen3-1.7B-Base-multilingual-tts-neucodec \
--bf16 --do_train --do_eval false --num_train_epochs 1 \
--train_file "multipacking" \
--logging_steps 2 \
--learning_rate 2e-5 \
--warmup_steps 0 \
--save_steps 100000 \
--save_total_limit 5 \
--gradient_checkpointing true \
--torch_dtype bfloat16 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 10 \
--remove_unused_columns false \
--torch-compile