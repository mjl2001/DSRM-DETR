# Dynamic Sampling and Reference Point Modulation for Enhanced Small Object Detection in UAV Imagery
📌 Important Note: This repository contains the official PyTorch implementation and guidelines for the manuscript currently submitted to The Visual Computer.This project is built upon the excellent MMDetection framework.

Overview
This repository provides the code, models, and usage documentation for DSRM-DETR, an enhanced deformable DETR architecture tailored for small object detection in Unmanned Aerial Vehicle (UAV) scenarios.

Detecting tiny, densely packed objects against complex aerial backgrounds remains a significant challenge. Traditional Transformer-based models often struggle with noisy sampling and inefficient feature aggregation. To address this, DSRM-DETR leverages a Swin Transformer backbone to capture robust multi-scale contextual features and introduces two core innovations:

Extensive experiments demonstrate that our approach achieves highly competitive performance, including a 24.9% mAP on the VisDrone2019 dataset (outperforming the baseline Deformable DETR by 1.7% mAP), balancing both detection accuracy and efficiency.
# Usage
It requires Python 3.7+, CUDA 9.2+ and PyTorch 1.5+.
# Step 1. Create a conda environment and activate it.
conda create --name x python=3.8 -y
conda activate x
# Step 2. Install PyTorch following official instructions, e.g.
On GPU platforms:
conda install pytorch torchvision -c pytorch

# Installation
pip install -r requirements

# Train
单GPU上训练：python train.py mmdetection-main/configs/dsrm-deformable_detr/dsrm-deformable-detr_r50_16xb2-50e_coco.py

数据集你需要自行到官网下载，VisDrone2019和CARPK
