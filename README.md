# Dynamic Sampling and Reference Point Modulation for Enhanced Small Object Detection in UAV Imagery

> **📌 Important Note:** This repository contains the official PyTorch implementation and guidelines for the manuscript currently submitted to ***The Visual Computer***. This project is built upon the excellent [MMDetection](https://github.com/open-mmlab/mmdetection) framework.

## Overview
This repository provides the code, models, and usage documentation for **DSRM-DETR**, an enhanced deformable DETR architecture tailored for small object detection in Unmanned Aerial Vehicle (UAV) scenarios.

Detecting tiny, densely packed objects against complex aerial backgrounds remains a significant challenge. Traditional Transformer-based models often struggle with noisy sampling and inefficient feature aggregation. To address this, DSRM-DETR leverages a Swin Transformer backbone to capture robust multi-scale contextual features and introduces two core innovations seamlessly integrated into the MMDetection framework:
* **Target-aware Reference Point Confidence Modulation (TPRCM):** Evaluates the reliability of reference points prior to feature aggregation, effectively suppressing background noise.
* **Dynamic Mask-based Sampling (DMS):** Adaptively allocates valid sampling points according to the object scale, optimizing computational resources.

Extensive experiments demonstrate that our approach achieves highly competitive performance, including a 24.9% mAP on the VisDrone2019 dataset (outperforming the baseline Deformable DETR by 1.7% mAP), balancing both detection accuracy and efficiency.

## Requirements
* Python: 3.7+ (3.8 is recommended)
* PyTorch: 1.5+ 
* CUDA: 9.2+

## Installation

**Step 1. Create a conda environment and activate it.**
```bash
conda create --name dsrm_detr python=3.8 -y
conda activate dsrm_detr

**Step 2.** Install PyTorch following the official instructions. For GPU platforms:
```bash
conda install pytorch torchvision -c pytorch

**Step 3.** Clone this repository and install the required dependencies.
```bash
git clone [https://github.com/your_username/DSRM-DETR.git](https://github.com/your_username/DSRM-DETR.git)
cd DSRM-DETR
pip install -r requirements.txt

Data Preparation
You need to download the VisDrone2019 and CARPK datasets from their official websites manually:
1、VisDrone Official Website
2、CARPK Official Website
After downloading, please organize the datasets in the data/ directory. Since MMDetection relies on the COCO format, we provide handy scripts to convert the original custom annotations to the standard COCO JSON format.

Training
Train on a single GPU:
```bash
python mmdetection-main/tools/train.py mmdetection-main/configs/dsrm-deformable_detr/dsrm-deformable-detr_r50_16xb2-50e_coco.py

Acknowledgement
This project is heavily based on MMDetection. We sincerely thank the OpenMMLab team for their outstanding contribution to the open-source computer vision community.
