# RecGRELA

This repository contains the reference code for the paper **Gated Rotary-Enhanced Linear Attention with Rank Modulation for Long-term Sequential Recommendation**.

## 1. Overall

<p align="center">
    <img src="img/overall.png" alt="overview_of_RecGRELA"/>
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

This repository contains the ML-1M dataset. If you want to train our model on other datasets, the LFM-1B, Tmall, and Netflix datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/188p9b-OkI1IZfg248AvkqFNiuHkRVdrJ?usp=sharing). ML-32M can also be found at [MovieLens](https://grouplens.org/datasets/movielens/) and processed by [conversion tools](https://github.com/RUCAIBox/RecDatasets/tree/master/conversion_tools).

### 4. Run

To reproduce the results reported in our paper, just run it:
```
python run_RecGRELA.py
```

### 5. Results
You can also check the training log in[`📁 log`](log/).


## Acknowledgment

Our code references [RecBole](https://github.com/RUCAIBox/RecBole), [Mamba4Rec](https://github.com/chengkai-liu/Mamba4Rec). 
