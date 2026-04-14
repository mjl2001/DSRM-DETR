# DSRM-DETR: Dynamic Sampling and Reference Point Modulation for Enhanced Small Object Detection in UAV Imagery
📌 Important Note: This repository contains the official PyTorch implementation and guidelines for the manuscript currently submitted to The Visual Computer.

Overview
This repository provides the code, models, and usage documentation for DSRM-DETR, an enhanced deformable DETR architecture tailored for small object detection in Unmanned Aerial Vehicle (UAV) scenarios.

Detecting tiny, densely packed objects against complex aerial backgrounds remains a significant challenge. Traditional Transformer-based models often struggle with noisy sampling and inefficient feature aggregation. To address this, DSRM-DETR leverages a Swin Transformer backbone to capture robust multi-scale contextual features and introduces two core innovations:

Target-aware Reference Point Confidence Modulation (TPRCM): Evaluates the reliability of reference points prior to feature aggregation, effectively suppressing background noise.

Dynamic Mask-based Sampling (DMS): Adaptively allocates valid sampling points according to the object scale, optimizing computational resources.

Extensive experiments demonstrate that our approach achieves highly competitive performance, including a 24.9% mAP on the VisDrone2019 dataset (outperforming the baseline Deformable DETR by 1.7% mAP), balancing both detection accuracy and efficiency.
