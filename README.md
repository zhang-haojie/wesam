<div align="center">

<h1> Improving the Generalization of Segmentation Foundation Model under Distribution Shift via Weakly Supervised Adaptation </h1>

<a href='https://zhang-haojie.github.io/project-pages/wesam.html'><img src='https://img.shields.io/badge/Project-Page-green'></a> 
<a href='http://arxiv.org/abs/2312.03502'><img src='https://img.shields.io/badge/Technique-Report-red'></a> 

</div>


## 🎈 News

- [2024.2.27] Our work has been accepted to CVPR 2024 🎉
- [2024.3.1] Training and inference code released

## 🚀 Introduction

<div align="center">
<img width="800" alt="image" src="asserts/teaser.webp?raw=true">
</div>

Segment Anything Model was pre-trained on a large-scale dataset but exhibits awkward performance on diverse downstream segmentation tasks. We adapt SAM through weak supervision to enhance its generalization capabilities.


## 📻 Overview

<div align="center">
<img width="800" alt="image" src="asserts/Pipeline.webp?raw=true">
</div>

The proposed self-training architecture with anchor network regularization and contrastive loss regularization. Red arrows indicates the backpropagation flow.


## 📆 TODO

- [x] Release code

## 🎮 Getting Started

### 1. Install Environment

see [INSTALL](INSTALL.md).

### 2. Prepare Dataset and Checkpoints

see [PREPARE](PREPARE.md).

### 3. Adapt with Weak Supervision

```
# 1 modify configs/config.py 
# Prompt type: box, point, coarse

# 2 adapt
python adaptation.py
```


## 🖼️ Visualization

<div align="center">
<img width="800" alt="image" src="asserts/VISUAL.webp?raw=true">
</div>


## 🎫 License

The content of this project itself is licensed under [LICENSE](LICENSE).

## 💡 Acknowledgement

- [SAM](https://github.com/facebookresearch/segment-anything)

- [lightning-sam](https://github.com/luca-medeiros/lightning-sam)

- [SAM-LoRA](https://github.com/JamesQFreeman/Sam_LoRA)

## 🖊️ Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@article{zhang2023improving,
      title={Improving the Generalization of Segmentation Foundation Model under Distribution Shift via Weakly Supervised Adaptation},
      author={Zhang, Haojie and Su, Yongyi and Xu, Xun and Jia, Kui},
      journal={arXiv preprint arXiv:2312.03502},
      year={2023}
}
```