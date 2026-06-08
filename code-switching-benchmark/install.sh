uv venv --python 3.12 --allow-existing

uv pip install huggingface_hub ipykernel datasets matplotlib vllm==0.19.1
uv pip install git+https://github.com/tchiayan/fastText.git

# download language identification model
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin