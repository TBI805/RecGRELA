# RecGRELA

This repository contains the reference code for the paper **Gated Rotary-Enhanced Linear Attention for Long-term Sequential Recommendation**. The paper is released on [arXiv](https://arxiv.org/abs/2506.13315).

## 1. Overall

<p align="center">
    <img src="img/RecGRELA.png" alt="overview_of_tim4rec"/>
  </p>

### 2. Requirements

Here are our main environment dependencies for running the repository：
- NVIDIA-SMI 535.183.01
- cuda 12.2
- python 3.11.5
- pytorch 2.4.0
- recbole 1.2.0
- casual-conv1d 1.4.0
- timm 1.0.11

### 3. Datasets

This repository contains the ML-1M dataset. If you want to train our model on other datasets, the ML-20M, ML-32M, and Netflix datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/188p9b-OkI1IZfg248AvkqFNiuHkRVdrJ?usp=sharing). ML-32M can also be found at [MovieLens](https://grouplens.org/datasets/movielens/) and processed by [conversion tools](https://github.com/RUCAIBox/RecDatasets/tree/master/conversion_tools).

### 4. Run

To reproduce the results reported in our paper, just run it:
```
python run_RecGRELA.py
```

### 5. Results
You can also check the training log in[`📁 log`](log/).

## 6. References
If you find this code useful or use the toolkit in your work, please consider citing:
```
@article{hu2025gatedrotaryenhancedlinearattention,
      title={Gated Rotary-Enhanced Linear Attention for Long-term Sequential Recommendation}, 
      author={Juntao Hu and Wei Zhou and Huayi Shen and Xiao Du and Jie Liao and Junhao Wen and Min Gao},
      journal={arXiv preprint arXiv:2506.13315},
      year={2025}
}
```

## Acknowledgment

Our code references [RecBole](https://github.com/RUCAIBox/RecBole), [Mamba4Rec](https://github.com/chengkai-liu/Mamba4Rec), and [Causal-Conv1d](https://github.com/Dao-AILab/causal-conv1d). We appreciate their outstanding work and open source. 
