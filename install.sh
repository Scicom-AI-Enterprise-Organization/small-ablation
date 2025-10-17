pip3 install torch==2.8.0
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip3 install flash-attn==2.8.3 --no-build-isolation
pip3 install wandb transformers accelerate
pip3 install mosaicml-streaming
pip3 install git+https://github.com/apple/ml-cross-entropy