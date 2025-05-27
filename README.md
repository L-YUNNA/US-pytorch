# US-pytorch

## 🧩 Overview

This project focuses on classifying tumors in ultrasound images through a two-stage pipeline: **Upstream** and **Downstream** tasks.

---

## 📌 Upstream Task

The upstream task involves self-supervised learning (SSL) using approximately 60,000 ultrasound images from publicly available datasets to pretrain models. Two different SSL approaches were employed:

- **DINOv1** using ResNet-50 as the backbone.
- **MAE (Masked Autoencoder)** using ViT-B/16 as the backbone.

These pretrained weights and associated training code can be found at the links below:

- [🔗 DINOv1 pretrained weights](https://github.com/L-YUNNA/US-pytorch)
- [🔗 MAE pretrained weights](https://github.com/L-YUNNA/US-pytorch)

**Datasets used in upstream training**:

| Organ       | Dataset                                                                                 | Target                                               | Num of data                        |
|-------------|------------------------------------------------------------------------------------------|------------------------------------------------------|------------------------------------|
| **Lung**    | COVIDx-US ([link](https://github.com/nrc-cnrc/COVID-US))                                                                    | COVID-19, Pneumonia, normal, other| 18,628 images<br>(188 videos)     |
| **Kidney**  | Open kidney dataset ([link](https://github.com/rsingla92/kidneyUS.git))                                                          | Segmentation of the kidney                          | 534 images                         |
| **Breast**  | BUSI ([link](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset))<br>Mendeley Data ([link](https://data.mendeley.com/datasets/wmy84gzngw/1))     | Benign, malignant, normal            | 780 images<br>250 images |
| **Heart**   | EchoNet-Dynamic Dataset ([link](https://echonet.github.io/dynamic/index.html))                                                      | Segmentation of the left ventricle                   | 40,120 images<br>(10,031 videos)   |                  |
| **Thyroid** | DDTI *(Thyroid ultrasound images)* ([link](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images))                                           | Segmentation of the thyroid nodules                 | 480 images                         |

> ✅ Total: More than 60,000 ultrasound images secured.

---

## 📌 Downstream Task

The downstream task focuses on tumor classification by transferring the pretrained weights from the upstream phase. Two models are used:

- A CNN-based classifier using **ResNet-50**
- A transformer-based classifier using **ViT-B/16**

The implementation for the downstream classification task is available in the repository:

---


