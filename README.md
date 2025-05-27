# US-pytorch

## Overview

This project focuses on classifying tumors in ultrasound images through a two-stage pipeline: **Upstream** and **Downstream** tasks.

---

## 🧩 Upstream Task

The upstream task involves self-supervised learning (SSL) using approximately 60,000 ultrasound images from publicly available datasets to pretrain models. Two different SSL approaches were employed:

- **DINOv1** using ResNet-50 as the backbone.
- **MAE (Masked Autoencoder)** using ViT-B/16 as the backbone.

These pretrained weights and associated training code can be found at the links below:

- [🔗 DINOv1 pretrained weights](#)
- [🔗 MAE pretrained weights](#)

**Datasets used in upstream training**:

| Organ       | Dataset                                                                                 | Target                                               | Num of data                        |
|-------------|------------------------------------------------------------------------------------------|------------------------------------------------------|------------------------------------|
| **Lung**    | COVIDx-US ([link](#))                                                                    | COVID-19, Pneumonia, normal, other| 18,628 images<br>(188 videos)     |
| **Kidney**  | Open kidney dataset ([link](#))                                                          | Segmentation of the kidney                          | 534 images                         |
| **Breast**  | BUSI ([link](#))<br>Mendeley Data ([link](#))     | Benign, malignant, normal            | 780 images<br>250 images |
| **Heart**   | EchoNet-Dynamic Dataset ([link](#))                                                      | Segmentation of the left ventricle                   | 40,120 images<br>(10,031 videos)   |                  |
| **Thyroid** | DDTI *(Thyroid ultrasound images)* ([link](#))                                           | Segmentation of the thyroid nodules                 | 480 images                         |

> ✅ Total: More than 60,000 ultrasound images secured.

---

## 📌 Downstream Task

The downstream task focuses on tumor classification by transferring the pretrained weights from the upstream phase. Two models are used:

- A CNN-based classifier using **ResNet-50**
- A transformer-based classifier using **ViT-B/16**

The implementation for the downstream classification task is available in the repository:

---


