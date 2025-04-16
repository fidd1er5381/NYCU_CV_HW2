# Digit Detection and Recognition

StudentID: 313553023 
Name: 褚敏匡

# NYCU Computer Vision 2025 Spring - Homework 2

## Introduction

This repository contains an implementation of a Faster R-CNN model with attention mechanisms for digit detection and recognition. The system can detect individual digits in images and recognize complete numbers formed by these digits, developed as part of the NYCU Computer Vision course.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/NYCU_CV_HW2.git
cd NYCU_CV_HW2
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Organize your dataset in the following structure:
```
data/
├── train/
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── valid/
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── test/
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── train.json  # COCO format annotations for training
└── valid.json  # COCO format annotations for validation
```

## Train and Predict

To train the model:
```bash
python train.py
```

To run prediction on test data:
```bash
python predict.py
```

This will generate:
- `pred.json`: Task 1 results (digit detection)
- `pred.csv`: Task 2 results (number recognition)
- Visualization of predictions on sample images

## Performance Snapshot

### Model Architecture
- Base Model: Faster R-CNN with ResNet-50-FPN
- Feature Enhancement: Convolutional Block Attention Module (CBAM)
- Anchor Sizes: (16, 32, 64, 128, 256)
- Aspect Ratios: (0.5, 1.0, 2.0)

### Training Configuration
- Optimizer: AdamW
- Learning Rate: 1e-4
- Weight Decay: 1e-4
- Learning Rate Scheduler: Cosine Annealing
- Batch Size: 8
- Early Stopping: Patience of 5 epochs

### Performance Metrics
- Task 1 (Digit Detection): mAP = 0.39
- Task 2 (Number Recognition): Accuracy = 0.80


## Outputs

- Best Model: `fasterrcnn_model_best.pth`
- Prediction Results: 
  - `pred.json`: COCO format detection results
  - `pred.csv`: Number recognition results
- Visualizations: `prediction_*.png`

## Attention Mechanism

The model integrates the Convolutional Block Attention Module (CBAM) which includes:
1. Channel Attention: Enhances important feature channels
2. Spatial Attention: Focuses on important spatial locations

This attention mechanism improves the model's ability to focus on relevant features for digit detection, particularly in complex backgrounds or when digits are densely arranged.

## Post-processing for Number Recognition

The system uses a sophisticated post-processing algorithm to combine detected digits into complete numbers:
1. DBSCAN clustering to group digits on the same line
2. Spatial arrangement analysis to order digits correctly
3. Digit combination to form complete numbers 
