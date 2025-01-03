# ToolGen Repository

## Introduction
This repository contains the official implementation of the following paper:

**(IROS 2024) Learning Generalizable Tool-use Skills through
Trajectory Generation**

Carl Qi*, Yilin Wu*, Lifan Yu, Haoyue Liu, Bowen Jiang, Xingyu Lin**, David Held**

[Website](https://sites.google.com/view/toolgen) /  [Paper](https://arxiv.org/abs/2310.00156)

## Usage
1. Initialize a conda environment (python=3.6) and install `python3 -m pip install -e .`
2. Install [torch (version 1.9.0 tested)](https://pytorch.org/get-started/previous-versions/)
    * We tested `pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html` on RTX 3090.
3. Install packages for computing the EMD loss:
    * [pykeops (1.5)](https://www.kernel-operations.io/keops/python/installation.html) by
      running `pip install pykeops==1.5`
    * [geomloss](https://www.kernel-operations.io/geomloss/api/install.html) by running `pip install geomloss`
4. Install necessary packages for [PointFlow](https://github.com/stevenygd/PointFlow)

5. (optional) Install **chester** from https://github.com/Xingyu-Lin/chester.

6. The training code for ToolGen has 2 parts: 
    * The first part is training the PointFlow model. The training script is in `PointFlow/scripts/gen_multitool.sh`
    * The second part is training the trajectory model. The training script code is in `core/toolgen/train_model.py`, and the model architecture and inference code is in `core/toolgen/bc_agent.py`.
7. The launch script that leverages chester to train/evaluate the ToolGen trajectory model is under `core/toolgen/launchers/launch_train_bc.py`. Alternatively, one can write a custom script calling run_task in `train_model.py` without using chester.

## Datasets
The training data from the ToolGen paper will be released soon, stay tuned!

## Cite

If you find this codebase useful in your research, please consider citing:

```
@INPROCEEDINGS{qi2024toolgen,
  author={Qi, Carl and Wu, Yilin and Yu, Lifan and Liu, Haoyue and Jiang, Bowen and Lin, Xingyu and Held, David},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Learning Generalizable Tool-use Skills through Trajectory Generation}, 
  year={2024},
  volume={},
  number={},
  pages={2847-2854},
  keywords={Point cloud compression;Deformable models;Shape;Autonomous systems;Affordances;Data models;Cleaning;Trajectory;Intelligent robots},
  doi={10.1109/IROS58592.2024.10801653}}
```

## Related Works
(CoRL 2022) Planning with Spatial-Temporal Abstraction from Point Clouds for Deformable Object Manipulation

Xingyu Lin*, Carl Qi*, Yunchu Zhang, Zhiao Huang, Katerina Fragkiadaki, Yunzhu Li, Chuang Gan, David Held

[Website](https://sites.google.com/view/pasta-plan) /  [Paper](https://arxiv.org/abs/2210.15751) / [Code](https://github.com/Xingyu-Lin/PASTA)

(RA-L 2022) Learning Closed-loop Dough Manipulation Using a Differentiable Reset Module

Carl Qi, Xingyu Lin, David Held

[Website](https://sites.google.com/view/dough-manipulation) /  [Paper](https://arxiv.org/pdf/2207.04638) [Code](https://github.com/carl-qi/diff_reset)

