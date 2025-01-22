# RecGLARE

## Usage


### Requirements

Here are our main environment dependencies for running the repository：
- NVIDIA-SMI 535.183.01
- cuda 12.2
- python 3.11.5
- pytorch 2.4.0
- recbole 1.2.0
- casual-conv1d 1.4.0
- timm 1.0.11

### DataSets

ML-20M, ML-32M, and Netflix datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/188p9b-OkI1IZfg248AvkqFNiuHkRVdrJ?usp=drive_link).

### Run

```
python run_RecGLARE.py
```


## Acknowledgment

Our code references [RecBole](https://github.com/RUCAIBox/RecBole), [Mamba4Rec](https://github.com/chengkai-liu/Mamba4Rec), and [Causal-Conv1d](https://github.com/Dao-AILab/causal-conv1d). We appreciate their outstanding work and open source. 
