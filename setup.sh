cd ..
git clone https://github.com/565353780/camera-control.git

cd camera-control
./setup.sh

pip install numpy pillow opencv-python scipy einops safetensors \
  trimesh matplotlib imageio tqdm requests onnxruntime \
  huggingface_hub evo mmengine transformers accelerate wandb \
  diffusers joblib decord iopath tensorboard hydra-core omegaconf

pip install viser==0.2.23
pip install gradio==5.17.1
pip install pydantic==2.10.6
