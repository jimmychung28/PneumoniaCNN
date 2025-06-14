# Pneumonia Detection CNN

A deep learning model for detecting pneumonia in chest X-ray images using Convolutional Neural Networks (CNN) with Keras/TensorFlow.

## Overview

This project implements a CNN model to classify chest X-ray images as either NORMAL or PNEUMONIA. The model addresses class imbalance in the dataset and includes various regularization techniques for improved performance.

## Dataset Structure

```
chest_xray/
├── train/
│   ├── NORMAL/      (1,349 images)
│   └── PNEUMONIA/   (3,883 images)
└── test/
    ├── NORMAL/      (234 images)
    └── PNEUMONIA/   (390 images)
```

**Note**: The dataset has a class imbalance with PNEUMONIA:NORMAL ratio of approximately 2.88:1 in the training set.

## Model Architecture

The CNN consists of 4 convolutional blocks with increasing filter sizes:
- Block 1: 32 filters → MaxPool → Dropout(0.25)
- Block 2: 64 filters → MaxPool → Dropout(0.25)
- Block 3: 128 filters → MaxPool → Dropout(0.25)
- Block 4: 256 filters → MaxPool → Dropout(0.25)

Followed by fully connected layers:
- Dense(512) → Dropout(0.5)
- Dense(256) → Dropout(0.5)
- Dense(1, sigmoid)

Each convolutional layer includes BatchNormalization for better training stability.

## Features

- **Class Weight Balancing**: Automatically calculates and applies class weights to handle dataset imbalance
- **Data Augmentation**: Rotation, shifts, zoom, and horizontal flips for better generalization
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Learning Rate Scheduling**: Reduces learning rate when validation loss plateaus
- **Model Checkpointing**: Saves the best model during training
- **Comprehensive Evaluation**: Provides accuracy, AUC, precision, recall, and confusion matrix
- **Visualization**: Generates training history plots and confusion matrix

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- tensorflow>=2.4.0
- keras>=2.4.0
- numpy>=1.19.2
- scikit-learn>=0.24.0
- matplotlib>=3.3.0
- seaborn>=0.11.0
- pillow>=8.0.0

## Usage

Run the training script:
```bash
python cnn.py
```

The script will:
1. Load and preprocess the chest X-ray images
2. Split training data into train/validation (80/20)
3. Train the model for 50 epochs with early stopping
4. Save the best model to `models/` directory
5. Generate evaluation metrics and visualizations
6. Save training logs to `logs/` directory for TensorBoard

## Output Files

After training, the following files will be generated:
- `models/pneumonia_cnn_best_[timestamp].h5` - Best model checkpoint
- `models/pneumonia_cnn_final_[timestamp].h5` - Final model
- `confusion_matrix.png` - Confusion matrix visualization
- `training_history.png` - Training metrics over epochs
- `logs/` - TensorBoard logs

## Model Performance Metrics

The model evaluates using:
- **Accuracy**: Overall classification accuracy
- **AUC**: Area Under the ROC Curve
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **Confusion Matrix**: Detailed breakdown of predictions

## TensorBoard Monitoring

To monitor training in real-time:
```bash
tensorboard --logdir=logs
```

Then open http://localhost:6006 in your browser.

## Customization

Key parameters can be adjusted in the `main()` function:
- `BATCH_SIZE`: Default 32
- `EPOCHS`: Default 50 (with early stopping)
- `INPUT_SHAPE`: Default (128, 128, 3)
- `learning_rate`: Default 0.0001

## License

This project is for educational and research purposes.