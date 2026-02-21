# Predictive Maintenance using LSTM (CMAPSS Dataset)

## Overview
This project implements a time-series deep learning model for Remaining Useful Life (RUL) prediction using the NASA CMAPSS FD001 turbofan engine dataset.

The objective is to predict the number of operational cycles remaining before engine failure using multivariate sensor data.

---

## Problem Statement
Given historical engine sensor readings, estimate the Remaining Useful Life (RUL) for each time step.

This is formulated as a supervised regression problem.

---

## Dataset
- Source: NASA CMAPSS FD001 dataset
- 100 training engines
- 21 sensor measurements + 3 operational settings
- Multivariate time-series data

---

## Methodology

### 1. Data Preprocessing
- Removed empty columns
- Generated RUL labels per engine
- Applied MinMax scaling
- Created sliding window sequences (sequence length = 30)
- Applied RUL clipping at 125 cycles to reduce variance

### 2. Model Architecture
- LSTM (64 units)
- Dropout (0.3)
- Dense (32, ReLU)
- Output layer (RUL prediction)

Loss: Mean Squared Error (MSE)  
Metric: Root Mean Squared Error (RMSE)

### 3. Training
- 80/20 train-validation split
- Early stopping (patience = 5)
- Fixed random seeds for reproducibility

---

## Results

Final Test RMSE: **~16.2 cycles**

Performance improved significantly after introducing RUL clipping (125), reducing extreme variance in high-RUL samples.

---

## Observed Failure Mode
The initial model systematically underpredicted high RUL values due to wide regression range and imbalance in early-life cycles.

Applying RUL clipping improved generalization and stability.

---

## Reproducibility

Seeds fixed for:
- NumPy
- Python random
- TensorFlow

Remaining nondeterminism may arise from low-level parallel GPU operations.

---

## Project Structure

```
predictive-maintenance/
│
├── data/ (excluded from repo)
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
python src/train.py
python src/evaluate.py
```

---

## License
MIT License