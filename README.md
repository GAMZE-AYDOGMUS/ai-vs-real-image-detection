# AI vs Real Image Detection

A deep learning based image classification project developed to distinguish AI-generated images from real images.

## Overview

This project uses transfer learning and fine-tuning with EfficientNet-B0 for binary image classification.

## Features

- AI-generated vs real image classification
- EfficientNet-B0 architecture
- Transfer learning and fine-tuning
- Model comparison with ResNet50 and DenseNet121
- Confusion matrix and classification report
- Gradio web interface
- Hugging Face Spaces compatible deployment

## Model Comparison

| Model | Accuracy | Training Time (sec) |
|---|---:|---:|
| ResNet50 | 93.45% | 2618 |
| DenseNet121 | 96.01% | 2693 |
| EfficientNet-B0 | 97.23% | 1330 |

Final fine-tuning validation accuracy: **98.62%**  
Test accuracy: **~99%**

## Dataset

The project uses AI-generated and real image datasets such as CIFAKE and GenImage.

Expected dataset structure:

```text
Islenmis_Veri/
├── train/
│   ├── FAKE/
│   └── REAL/
├── val/
│   ├── FAKE/
│   └── REAL/
└── test/
    ├── FAKE/
    └── REAL/
```

## Installation

```bash
pip install -r requirements.txt
```

## Run Gradio App

Place the trained model file at:

```text
models/best_model.pt
```

Then run:

```bash
python app.py
```

## Repository Structure

```text
ai_vs_real_image_detection_github_clean.ipynb
app.py
requirements.txt
models/
images/
results/
```

## Notes

Large datasets and trained model files are not included in the repository. Use Git LFS or provide an external download link for model weights.
