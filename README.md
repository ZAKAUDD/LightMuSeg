# LightMuSeg

## Lightweight Multiscale Feature Refinement Network for Breast Cancer Histopathology Segmentation

<p align="center">
  <img src="figures/LightMuSeg_architecture.png" alt="LightMuSeg Architecture" width="900">
</p>

<p align="center">
  <b>A lightweight and efficient deep learning framework for breast cancer histopathology image segmentation</b>
</p>

---

## 📌 Overview

Breast cancer is one of the most common cancers affecting women worldwide. Histopathological image analysis plays an important role in breast cancer diagnosis because it provides detailed information at the tissue and cellular levels.

With the increasing adoption of digital pathology, histopathological Whole Slide Images (WSIs) can reach gigapixel resolution. Automatically segmenting relevant tissue structures from these large images is therefore an important but computationally demanding task.

High-performing segmentation models often contain millions of parameters and may require substantial computational resources, making them difficult to deploy in real-world clinical and high-throughput pathology workflows.

To address these challenges, we propose **LightMuSeg**, a lightweight multiscale feature refinement network for breast cancer histopathology segmentation.

LightMuSeg is designed around three main objectives:

- **High segmentation accuracy**
- **Fast inference**
- **Low computational and memory requirements**

The proposed framework uses a **pre-trained MobileNet-V2 encoder** for lightweight feature extraction and introduces two specialized modules:

1. **Multi-Scale Feature Fusion Module (MSFFM)**
2. **Feature Refinement and Fusion Module (FRFM)**

MSFFM extracts contextual information using multiple receptive fields to handle the large scale variation of tissue structures. FRFM refines and fuses encoder and decoder features while reducing redundant feature responses and representation gaps introduced by upsampling and skip connections.

The model is evaluated on:

- **BCSS** for binary segmentation
- **BCSS-WSSS** for weakly supervised multiclass segmentation

---

## 📄 Paper

**Zaka-Ud-Din Muhammad and Vincenzo Della Mea**

> **Lightweight multiscale feature refinement network for breast cancer histopathology segmentation**

*Informatics in Medicine Unlocked*, Volume 63, 101754, 2026.

**DOI:** :contentReference[oaicite:0]{index=0}

**Published:** 20 April 2026

**Received:** 23 October 2025  
**Revised:** 9 April 2026  
**Accepted:** 12 April 2026

The article is published under the **CC BY-NC 4.0** license.

---

# 🧠 Motivation

Deep learning architectures such as U-Net have substantially improved medical image segmentation. However, histopathological WSI segmentation presents several challenges:

- Very high image resolution
- Large variation in tissue and object size
- Different cellular and tissue morphologies
- Indistinct object boundaries
- Heterogeneous staining
- Camouflaged structures
- Complex background regions
- Limited annotated WSI data
- High computational requirements

A model intended for real-world digital pathology should therefore not only achieve high Dice and IoU scores, but should also provide:

```text
                ┌────────────────────┐
                │ High Accuracy      │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Fast Inference     │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Small Model Size   │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Clinical / WSI     │
                │ Deployment         │
                └────────────────────┘

@article{muhammad2026lightmuseg,
  title={Lightweight multiscale feature refinement network for breast cancer histopathology segmentation},
  author={Muhammad, Zaka-Ud-Din and Della Mea, Vincenzo},
  journal={Informatics in Medicine Unlocked},
  volume={63},
  pages={101754},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.imu.2026.101754}
}
