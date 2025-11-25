curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 --allow-existing
source .venv/bin/activate
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
FLASH_ATTENTION_SKIP_CUDA_BUILD="TRUE" uv pip install flash-attn==2.8.3 --no-build-isolation
uv pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280 --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install wandb transformers==4.57.1 accelerate mosaicml-streaming liger_kernel hf-transfer