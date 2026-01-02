# 3D Brain Tumor Segmentation (HelioGLI)

## Project Overview
This solution implements an automated 3D semantic segmentation pipeline for brain tumors using the **MONAI** framework and **PyTorch**. The core architecture is **SegResNet**, a 3D ResNet-based encoder-decoder network optimized for medical image segmentation.

## Visualizations

### 1. Multi-Planar Orthogonal View
We visualize the tumor from three standard medical perspectives (Axial, Coronal, Sagittal) to confirm 3D spatial consistency. The **AI Prediction (Red)** aligns precisely with the **Ground Truth (Green)** across all planes.

![Orthogonal View Result](Images/Multimodal_Result.png)

### 2. 3D Volumetric Segmentation
The following animation demonstrates the model's performance across the 3D volume (Slices 50–110). The sliding window inference strategy ensures consistent boundary predictions.

![Volumetric Segmentation Animation](Images/Multimodal_Video.gif)

## Model Architecture
* **Architecture:** SegResNet (Spatial Dimensions: 3)
* **Input Channels:** 4 (T1n, T1c, T2w, T2f)
* **Output Channels:** 3 (Tumor Classes)
* **Preprocessing:** Orientation normalization (RAS), isotropic respacing (1mm), and Z-score intensity normalization.
* **Inference Strategy:** Sliding Window Inference (Window size: $128 \times 128 \times 128$) with 50% overlap.

##  Training Configuration

The model was trained on the **HelioGLI** dataset using a 3D SegResNet architecture. Below are the specific hyperparameters and settings used during the training phase.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Architecture** | 3D SegResNet | ResNet-based encoder-decoder with variational autoencoder regularization. |
| **Input Shape** | `(4, 160, 192, 128)` | 4 MRI Modalities (T1n, T1c, T2w, T2f) cropped to 3D patches. |
| **Loss Function** | Dice Loss | Optimized for volumetric segmentation overlap (handles class imbalance). |
| **Optimizer** | Adam | Adaptive Moment Estimation with weight decay `1e-5`. |
| **Learning Rate** | `1e-4` | Initial learning rate with Cosine Annealing scheduler. |
| **Batch Size** | 1 | Optimized for single GPU training with large 3D volumes. |
| **Epochs** | 20 | 
| **Framework** | MONAI / PyTorch | Medical Open Network for AI. |

## Installation
Ensure you have Python installed, then install the dependencies:

```bash
pip install -r requirements.txt
