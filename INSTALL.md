## Installation


### Requirements
- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 1.13.1 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
- Install pytorch [lightning](https://lightning.ai/pytorch-lightning) that matches the PyTorch installation.
- `pip install -r requirements.txt`


### Example conda environment setup
```bash
conda create --name wesam python=3.8
conda activate wesam

# CUDA 11.7
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

git clone https://github.com/zhang-haojie/wesam.git
cd wesam
pip install -r requirements.txt
```