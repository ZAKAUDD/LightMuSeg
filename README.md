# Lightweight Multiscale Feature Refinement Network for Breast Cancer Histopathology Segmentation (LightMuSeg)

A **lightweight and efficient deep learning framework for breast cancer histopathology image segmentation**, designed to balance segmentation performance, inference speed, and model efficiency for large-scale Whole Slide Image (WSI) analysis.

## 📌 Overview

Breast cancer is one of the most prevalent cancers affecting women worldwide. With the increasing adoption of digital pathology, histopathological **Whole Slide Images (WSIs)** can reach gigapixel resolution, making automated tissue segmentation both important and computationally demanding.

While high-capacity segmentation architectures can achieve strong performance, their computational and memory requirements can make them difficult to deploy in real-world clinical and high-throughput pathology workflows.

To address this challenge, we propose **LightMuSeg (Lightweight Multiscale Feature Refinement Network)**, a compact segmentation framework designed around three objectives:

* **High segmentation performance**
* **Fast inference**
* **Low computational and memory requirements**

The proposed architecture combines a **pre-trained MobileNet-V2 encoder** with two specialized modules:

1. **Multi-Scale Feature Fusion Module (MSFFM)** – captures contextual information across multiple scales to address the substantial variation in tissue structures.
2. **Feature Refinement and Fusion Module (FRFM)** – refines and fuses encoder and decoder representations, reducing redundant feature responses and representation gaps introduced by upsampling and skip connections.

Together, these components enable LightMuSeg to capture both fine-grained and contextual tissue information while maintaining a lightweight architecture.

---

## 🧠 Motivation

Histopathological WSI segmentation presents several challenges:

* Gigapixel-scale image resolution
* Large variation in tissue and object sizes
* Heterogeneous tissue and cellular morphology
* Indistinct object boundaries
* Staining variability
* Camouflaged tissue structures
* Complex background regions
* Limited annotated data
* High computational requirements

For practical digital pathology applications, segmentation models should therefore consider not only **Dice and IoU**, but also **inference efficiency and model size**.

LightMuSeg addresses these requirements by combining multiscale feature extraction and feature refinement with an efficient MobileNet-V2 backbone.

---

## 🏗️ Architecture

LightMuSeg consists of three main components:

### 1. MobileNet-V2 Encoder

A **pre-trained MobileNet-V2** is used as the encoder to provide efficient feature extraction while significantly reducing the number of parameters compared with conventional heavyweight backbones.

### 2. Multi-Scale Feature Fusion Module (MSFFM)

The **MSFFM** captures contextual information at multiple receptive fields, allowing the network to better represent tissue structures with different spatial scales and morphological characteristics.

### 3. Feature Refinement and Fusion Module (FRFM)

The **FRFM** enhances discriminative features by refining and fusing encoder and decoder representations. It helps reduce redundant feature responses and representation gaps caused by upsampling and skip connections, particularly for challenging and camouflaged structures.

---

## 🔬 Dataset and Evaluation

LightMuSeg is evaluated on two breast cancer histopathology segmentation benchmarks:

### BCSS

The **Breast Cancer Semantic Segmentation (BCSS)** dataset is used for binary segmentation evaluation.

### BCSS-WSSS

The **BCSS-WSSS** dataset is used to evaluate weakly supervised multiclass segmentation performance.

The evaluation covers both binary and multiclass segmentation scenarios, providing a broader assessment of the model's ability to generalize across different segmentation settings.

---

## 🔍 Patient-Level Data Partitioning

An important consideration when working with WSIs is **data leakage and distribution bias**.

WSIs are often divided into multiple tiles extracted from Regions of Interest (ROIs). If these tiles are randomly divided between training and test sets, highly correlated samples originating from the same patient can appear in both sets. This can lead to overly optimistic performance estimates and does not accurately represent real-world deployment.

To mitigate this issue, LightMuSeg uses **patient-level (case-level) data partitioning**.

This ensures that:

* Patients in the test set are completely unseen during training.
* Correlated tiles from the same patient do not appear across training and test sets.
* Evaluation better reflects generalization to unseen patients.
* Reported performance is more representative of real-world clinical deployment.

This evaluation strategy is particularly important for WSI-based computational pathology, where a single patient may contribute a large number of highly correlated image tiles.

---

## 📊 Key Results

LightMuSeg achieves a strong balance between segmentation performance and computational efficiency:

| Metric              |             LightMuSeg |
| ------------------- | ---------------------: |
| **Dice**            |             **76.51%** |
| **IoU**             |             **66.43%** |
| **Inference Speed** | **44.51 tiles/second** |
| **Parameters**      |              **7.05M** |

The model demonstrates strong segmentation performance on important tissue classes, including **lymphocytic infiltrate** and **necrosis**, while maintaining a compact architecture.

### ⚡ Efficiency

The lightweight design makes LightMuSeg particularly relevant for high-throughput digital pathology environments, where large numbers of WSIs may need to be processed efficiently.

For example, pathology laboratories may process hundreds or thousands of slides per day. In such settings, reducing model size and inference time can be important for scalable deployment.

---

## 📈 Main Contributions

The main contributions of this work are:

* A **lightweight multiscale feature refinement network** for breast cancer histopathology segmentation.
* Use of a **pre-trained MobileNet-V2 encoder** to reduce computational complexity.
* A **Multi-Scale Feature Fusion Module (MSFFM)** for capturing tissue structures across different spatial scales.
* A **Feature Refinement and Fusion Module (FRFM)** for enhancing discriminative representations and addressing challenging structures.
* Evaluation on **BCSS** and **BCSS-WSSS** for binary and weakly supervised multiclass segmentation.
* **Patient-level data partitioning** to reduce potential data leakage and provide a more realistic evaluation of generalization.
* A favorable balance between **segmentation accuracy, inference speed, and model size**.

---

## 📄 Publication

**Zaka-Ud-Din Muhammad and Vincenzo Della Mea**

> **Lightweight multiscale feature refinement network for breast cancer histopathology segmentation**

*Informatics in Medicine Unlocked*, Volume 63, Article 101754, 2026.

**Published:** 20 April 2026
**Accepted:** 12 April 2026
**Revised:** 9 April 2026
**Received:** 23 October 2025

**DOI:** `10.1016/j.imu.2026.101754`

The article is published under the **CC BY-NC 4.0** license.

---

## 📚 Citation

If you use LightMuSeg in your research, please cite:

```bibtex
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
```

---

## 🤝 Acknowledgements

We would like to thank **Vincenzo Della Mea** for his supervision, guidance, and support throughout this research.

This work was made publicly available through the **BosomShield** project.

---

## 📌 Summary

**LightMuSeg** provides a lightweight solution for breast cancer histopathology segmentation by combining:

**MobileNet-V2 + Multi-Scale Feature Fusion + Feature Refinement & Fusion**

The resulting framework achieves:

> **76.51% Dice | 66.43% IoU | 44.51 tiles/s | 7.05M parameters**

while using patient-level evaluation to provide a more realistic assessment of generalization to unseen cases.

---
