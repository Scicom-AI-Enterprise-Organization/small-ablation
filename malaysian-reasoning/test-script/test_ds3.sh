CUDA_HOME="/usr/local/cuda-12.8" \
TORCH_DISTRIBUTED_DEBUG="info" \
deepspeed test_ds3.py \
--model_name_or_path unsloth/gpt-oss-120b-BF16 \
--train_file "malaysian-reasoning-8k-mosaic" \
--model_dtype bfloat16 \
--rank 16 \
--alpha 32