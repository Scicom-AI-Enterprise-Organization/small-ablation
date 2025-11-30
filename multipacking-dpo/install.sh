curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 --allow-existing
source .venv/bin/activate
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch290 --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install transformers==4.57.1 datasets accelerate
uv pip install mosaicml-streaming
uv pip install wandb
uv pip install liger-kernel
uv pip install ipython ipykernel
ipython kernel install --user --name=multipacking-dpo