# FlowRefinerPlus

This repository contains the code for the paper: Fighting the Enemy Within: Traffic Semantics-Driven Framework Against Noisy Labels for Robust Traffic Classification (IEEE Transactions on Networking).

## Setup

Dependencies:
- Python ≥ 3.8
- PyTorch ≥ 1.10
- torchvision
- timm==0.3.2
- scikit-learn
- matplotlib

## Dataset

The dataset format is consistent with the [YaTC](https://github.com/NSSL-SJTU/YaTC):

```
[Dataset_Path]/[train/test]/[Class]/[Sample]
```

Noisy dataset example (ISCXVPN with 20% label noise): [Link](https://drive.google.com/file/d/18pOj6YQZntyBvEPYYYRi3x8ULdtQN7qM/view?usp=sharing)

```
Path: ./data/ISCXVPN_noisy20
```

To generate a dataset with a specified label noise ratio:

```
Code: data_process.py
Function: make_noise_dataset(raw_dataset_path, noise_ratio)
```

## Run

Pre-trained encoder: [Link](https://drive.google.com/file/d/1wWmZN87NgwujSd2-o5nm3HaQUIzWlv16/view?usp=sharing)

```
Path: ./output_dir/pretrained_encoder.pth
```

Run FlowRefinerPlus:

```
python FlowRefiner_train.py 
```

## Contact-Info

Link to our laboratory: [SJTU-NSSL](https://github.com/NSSL-SJTU)

[Mingwei Zhan](https://github.com/zmw1216)
Email: [mw.zhan@sjtu.edu.cn](mailto:mw.zhan@sjtu.edu.cn)

[Ruijie Zhao](https://github.com/iZRJ)
Email: [ruijiezhao@seu.edu.cn](mailto:ruijiezhao@seu.edu.cn)
