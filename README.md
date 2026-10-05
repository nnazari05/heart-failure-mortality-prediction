# Heart Failure Mortality Prediction with PyTorch

A **PyTorch-based Multilayer Perceptron (MLP)** for predicting mortality events in patients with heart failure.

## Overview

This project uses the **Heart Failure Clinical Records Dataset** to perform binary classification of the `DEATH_EVENT` target using 12 clinical features.

The workflow covers data preprocessing, feature standardization, stratified dataset splitting, model training, and validation.

## Target Labels

* `0` — Patient survived during the follow-up period
* `1` — Patient died during the follow-up period

## Model Architecture

The model is a fully connected **Multilayer Perceptron (MLP)** consisting of two hidden layers with ReLU activations:

```text
Input (12)
   ↓
Linear (12 → 32)
   ↓
ReLU
   ↓
Linear (32 → 16)
   ↓
ReLU
   ↓
Linear (16 → 2)
   ↓
Output
```

The model is trained using **Cross-Entropy Loss** and the **SGD** optimizer.
