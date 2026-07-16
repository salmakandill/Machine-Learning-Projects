# Bank Marketing Prediction

## Goal
Predict whether the customer will subscribe to a term deposit.

## Dataset
- Rows: 45,211
- Features: 16
- Target: y

## Preprocessing
- Label Encoding:
  - yes → 1
  - no  → 0
- One-Hot Encoding:
  - job
  - marital
  - education
  - contact
  - month
  - poutcome
- StandardScaler

## Model
- Logistic Regression
- max_iter = 1000
- class_weight = balanced

## Results
Accuracy : 84%
Precision: 42%
Recall   : 83%
F1 Score : 56%

## What I Learned
- One-Hot Encoding
- StandardScaler
- Confusion Matrix
- Imbalanced Dataset

## Notes
- Dataset is imbalanced.
- Using class_weight='balanced' improved Recall.