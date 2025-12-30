# 3D Brain Tumor Segmentation (HelioGLI)

## Project Overview
This solution implements an automated 3D semantic segmentation pipeline for brain tumors using the **MONAI** framework and **PyTorch**. The core architecture is **SegResNet**, a 3D ResNet-based encoder-decoder network optimized for medical image segmentation.

## Model Architecture
* **Architecture:** SegResNet (Spatial Dimensions: 3)
* **Input Channels:** 4 (T1n, T1c, T2w, T2f)
* **Output Channels:** 3 (Tumor Classes)
* **Preprocessing:** Orientation normalization (RAS), isotropic respacing (1mm), and intensity normalization.
* **Inference Strategy:** Sliding Window Inference (Window size: 128x128x128) with 50% overlap for consistent boundary predictions.

## Installation
Ensure you have Python installed, then install the dependencies:

```bash
pip install -r requirements.txt
