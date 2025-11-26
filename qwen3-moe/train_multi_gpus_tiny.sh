# e.g., use 2 GPUs on one machine
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=8

torchrun --nproc_per_node=2 --master_port=23456 example_train_tiny_multi_gpus.py
