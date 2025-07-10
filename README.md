# Sensor-Based-Human-Activity-Classification-Using-Deep-Learning
This project applies machine learning techniques to classify activity using Passive Infrared (PIR) sensor data along with temperature readings. It incorporates data preprocessing, model training using MLP and LSTM architectures (implemented in PyTorch), and model checkpointing for subsequent evaluations.

---

## Table of Contents

- [Overview](#overview)
- [Key Libraries](#key-libraries)
- [Preprocessing Steps](#preprocessing-steps)
- [Models](#models)
- [Workflow](#workflow)
- [Function Summaries](#function-summaries)
  - [MLP Training & Evaluation](#mlp-training--evaluation)
  - [LSTM Training & Evaluation](#lstm-training--evaluation)
- [Output Files](#output-files)
- [Model Evaluation on Testing Data](#model-evaluation-on-testing-data)
  - [Independent Evaluation Instructions](#independent-evaluation-instructions)
  - [Function: `evaluate_model`](#function-evaluate_model)
- [Optional Parameters](#optional-parameters)

---

## Overview

The pipeline involves:
- **Data Preprocessing:** Cleansing, scaling, outlier removal, and oversampling using SMOTE.
- **Model Training:** Utilizing both MLP for aggregated features and LSTM for sequence learning.
- **Cross-Validation:** Using stratified 5-Fold Cross-Validation coupled with early stopping to mitigate overfitting.
- **Model Checkpointing:** Saving the best performing model on each fold.

---

## Key Libraries

The project leverages the following essential libraries:

```python
import gdown
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
```
## Preprocessing Steps

- **Data Loading:** Import PIR sensor readings, temperature (Temperature_F), and labels.
- **Pattern Analysis:** Evaluate the label distribution against different temperature values.
  - **Note:** Label 3 appears only when Temperature_F is 0, and a temperature of 0 is distinctly an outlier.
- **Outlier Removal:** Exclude entries based on raw temperature differences.
- **SMOTE Application:** Address class imbalance by oversampling.
- **Feature Normalization:** Standardize features using `StandardScaler`.
- **Data Structuring:** Prepare datasets for two different modes:
  - **MLP:** Flattened feature vectors.
  - **LSTM:** Sequential PIR sensor readings along with per-timestep temperature features.
## Models

- **MLP (Multi-Layer Perceptron):**  
  Designed for classification using aggregated features, the MLP model processes flattened PIR sensor data and temperature inputs. It includes:
  - Cross-validation for robust performance estimation.
  - Early stopping to prevent overfitting.
  - Checkpointing to save the best model based on validation accuracy.

- **LSTM (Long Short-Term Memory):**  
  Crafted for sequential data, the LSTM model takes in time-series PIR sensor readings along with temperature at each timestep. It incorporates:
  - Sequence-based learning for capturing temporal dynamics.
  - Early stopping and cross-validation similar to the MLP approach.
  - Checkpointing to retain the model with the highest validation performance.
## Workflow

1. **Data Loading and Preprocessing:**
   - Import PIR sensor readings, temperature, and labels.
   - Analyze patterns and eliminate outliers.
   - Apply SMOTE to balance the dataset.
   - Normalize the features using `StandardScaler`.

2. **Data Structuring:**
   - Format data for two modes:
     - **MLP:** Use flattened feature vectors.
     - **LSTM:** Use sequential PIR sensor data along with per-timestep temperature features.

3. **Cross-Validation Loop:**
   - Use stratified 5-Fold Cross-Validation for robust model training and validation.
   - Implement early stopping to avoid overfitting.

4. **Model Training and Checkpointing:**
   - Train the MLP and LSTM models for each fold.
   - Save the best-performing model based on validation accuracy for later evaluation.

5. **Evaluation:**
   - Assess model performance using metrics such as accuracy and macro F1 score.
   - Generate confusion matrices and classification reports.
## Function Summaries

### MLP Training & Evaluation
- **`train_and_validate_mlp_best_only(X, y, raw_temp, num_classes, checkpoint_path='team_6_checkpoints/team_6_mlp.pt')`**
  - Trains an MLP model using 5-fold cross-validation.
  - Implements early stopping based on validation performance to prevent overfitting.
  - Saves the best model checkpoint for each fold.
  - Returns metrics such as accuracy and macro F1 score for each fold.

- **`evaluate_mlp(checkpoint_path, X_test, y_test, temp_test_raw, input_dim, num_classes)`**
  - Loads the best saved MLP model.
  - Evaluates the model on test data by computing test accuracy, confusion matrix, and generating a classification report that includes the macro F1 score.

### LSTM Training & Evaluation
- **`train_and_validate_lstm_best_only(X_seq, y, raw_temp, num_classes, checkpoint_path='team_6_checkpoints/team_6_lstm.pt')`**
  - Trains an LSTM model using 5-fold cross-validation on sequential data.
  - Incorporates early stopping to optimize training and mitigate overfitting.
  - Saves the best model checkpoint based on validation performance.
  - Returns detailed training metrics including losses and accuracies for each fold.

- **`evaluate_lstm(checkpoint_path, X_test_seq, y_test, temp_test_raw, num_classes)`**
  - Loads the best saved LSTM model from the specified checkpoint.
  - Evaluates the model on sequential test data.
  - Computes test accuracy and produces a confusion matrix and classification report including macro F1 score.
  - Displays a heatmap of the confusion matrix for visual performance analysis.
## Output Files

- `team_6_checkpoints/team_6_mlp.pth` – Checkpoint for the best MLP model per fold.
- `team_6_checkpoints/team_6_lstm.pth` – Checkpoint for the best LSTM model per fold.
- Additional outputs such as training logs, confusion matrices, and plots may also be generated.
## Model Evaluation on Testing Data

The final performance of the models is assessed using independent test data. This process involves:
- Loading the best saved model.
- Preprocessing test datasets using the same procedures as training.
- Evaluating metrics such as classification accuracy, confusion matrix, and macro F1 score.
### Independent Evaluation Instructions

To conduct an independent evaluation:
1. Open the `.ipynb` file (e.g., `team_6.ipynb`).
2. Execute the last two cells to:
   - Load and preprocess the test dataset.
   - Load the saved model checkpoint.
   - Evaluate the model and display performance metrics along with visual outputs (e.g., confusion matrix heatmap).
### Function: `evaluate_model`

The `evaluate_model(datafile_path, checkpoint_path)` function is responsible for testing the saved LSTM model. It works as follows:
- **Data Loading & Preprocessing:** Reads test data from a CSV file, remaps labels (e.g., converting label `3` to `2` for consistency), scales the PIR sensor features, and formats the data into tensors.
- **Model Evaluation:** Loads the saved model checkpoint and performs inference on the test data.
- **Output:** Returns the model's classification accuracy on the test dataset.

**Parameters:**
- `datafile_path` *(str)*: The path to the CSV file with test data.
- `checkpoint_path` *(str)*: The path to the saved PyTorch model checkpoint (`.pt` or `.pth`).

**Returns:**
- `acc` *(float)*: The accuracy of the model on the test dataset.

**Example:**
```python
acc = evaluate_model("data/test_fold_0.csv", "checkpoints/best_lstm_model_fold0.pt")
print(f"Test Accuracy: {acc:.4f}")
```  
### Optional Parameters

```md
## Optional Parameters

- `--epochs`: Number of training epochs (default: 50).
- `--batch_size`: Batch size for training (default: 64).
- `--folds`: Number of cross-validation folds (default: 5).

