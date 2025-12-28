# Brain Tumor MRI Classification -- End-to-End Deep Learning System

**Author:** Lucas Kpatah\
**Date:** December 28, 2025

------------------------------------------------------------------------

## Overview

This project implements a **high-accuracy deep learning pipeline** for
classifying brain MRI scans into four clinically relevant categories:

-   **Glioma**
-   **Meningioma**
-   **Pituitary tumor**
-   **No tumor (Normal)**

The system covers the **entire ML lifecycle**: data preparation →
training → evaluation → explainability → deployment as a containerized
inference API.

No frontend UI is included by design. The FastAPI Swagger UI (`/docs`)
serves as the testing and interaction interface.

------------------------------------------------------------------------

## Objective

-   Train a state-of-the-art CNN using transfer learning
-   Achieve near-perfect classification accuracy
-   Provide **explainable predictions** via Grad-CAM
-   Package everything into a **lightweight Docker image** for
    reproducible inference

------------------------------------------------------------------------

## Dataset

-   Public Brain Tumor MRI dataset (Kaggle / Figshare variants)

-   Folder-based structure:

        dataset/
        ├── Training/
        │   ├── glioma/
        │   ├── meningioma/
        │   ├── pituitary/
        │   └── notumor/
        └── Testing/
            ├── glioma/
            ├── meningioma/
            ├── pituitary/
            └── notumor/

-   Several thousand MRI slices

-   Images mostly grayscale but stored as RGB

-   Slight class imbalance (acceptable)

------------------------------------------------------------------------

## Exploratory Data Analysis (EDA)

-   Loaded with `torchvision.datasets.ImageFolder`
-   Verified class counts
-   Visualized representative samples from each class
-   Confirmed visually distinguishable tumor patterns

------------------------------------------------------------------------

## Preprocessing & Augmentation

**Input size:** 224×224

**Training transforms:** - Resize - Random horizontal flip (p = 0.5) -
Random rotation (±15°) - RandomAffine (small translation) - Normalize
with ImageNet statistics

**Normalization:**

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

Same normalization used for training and evaluation.

------------------------------------------------------------------------

## Model Architecture

-   **EfficientNet-B0** (via `timm`)
-   Pretrained on ImageNet
-   Final classifier replaced for 4-class output

**Why EfficientNet-B0?** - Excellent accuracy-to-size ratio - Proven
performance in medical imaging - Efficient inference on CPU

------------------------------------------------------------------------

## Training Setup

-   Optimizer: **AdamW**
-   Learning rate: 0.001
-   Weight decay: 0.01
-   Loss: CrossEntropyLoss
-   Scheduler: ReduceLROnPlateau
-   Epochs: 15
-   Batch size: 32--64 (GPU dependent)

------------------------------------------------------------------------

## Results

-   **Best validation accuracy:** 99.93% (epoch 9)
-   **Final validation accuracy:** 99.47%
-   Minimal overfitting
-   Precision / Recall / F1 ≈ 1.0 for all classes
-   Confusion matrix showed only rare misclassifications

------------------------------------------------------------------------

## Explainability (Grad-CAM)

-   Grad-CAM applied to: `model.blocks[-1][-1].conv_pw`
-   Heatmaps correctly highlight tumor regions
-   Clinically interpretable explanations
-   Available via API endpoint

------------------------------------------------------------------------

## FastAPI Inference Service

### Endpoints

-   **POST `/predict`**
    -   Input: MRI image
    -   Output: JSON prediction, confidence, class probabilities
-   **POST `/predict/visualize`**
    -   Input: MRI image
    -   Output: PNG with Grad-CAM overlay

Swagger UI:

    http://localhost:8000/docs

------------------------------------------------------------------------

## Docker Containerization

### Optimization Strategy

-   Base image: `python:3.9-slim`
-   CPU-only PyTorch wheels
-   Minimal system dependencies
-   Removed training-only libraries

### Final Image Size

-   **1.9 GB** (down from 12+ GB)

------------------------------------------------------------------------

## Project Structure

    Brain-Tumor-DT/
    ├── api.py
    ├── best_brain_tumor_model.pth
    ├── requirements.txt
    ├── Dockerfile
    ├── .dockerignore
    └── README.md

------------------------------------------------------------------------

## Running the Project

### Build Docker Image

    docker build -t brain-tumor-api:small .

### Run Container

    docker run -p 8000:8000 brain-tumor-api:small

### Test

Open:

    http://localhost:8000/docs

------------------------------------------------------------------------

## Reproducibility Notes

-   Deterministic preprocessing
-   Standard ImageNet normalization
-   CPU-only inference supported
-   Docker ensures environment consistency


## Conclusion

This project demonstrates a **complete, production-quality medical
imaging pipeline** with:

-   State-of-the-art accuracy (99.93%)
-   Explainable AI via Grad-CAM
-   Highly optimized deployment (1.9 GB Docker image)
-   Clean, reproducible, and professional engineering

