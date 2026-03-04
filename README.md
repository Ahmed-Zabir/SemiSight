## SemiSight — Semiconductor Fault Prediction Platform
An end-to-end machine learning platform for semiconductor yield analytics, built on the UCI SECOM and WM-811K datasets.

---

## Project Overview

SemiSight predicts semiconductor chip pass/fail outcomes, explains model decisions using SHAP, investigates failure clustering, and classifies wafer map failure patterns using a CNN — all deployed in an interactive Streamlit dashboard.

---

## Phases

| 1 | EDA & Data Cleaning
| 2 | Preprocessing & Feature Engineering
| 3 | Model Building & Evaluation
| 4 | SHAP Explainability
| 5 | Failure Mode Clustering 
| 6 | CNN Wafer Map Analysis 
| 7 | Streamlit Dashboard 

---

## Key Results

- **Random Forest** champion model — AUC-ROC 0.768, Pass F1 0.32 on severely imbalanced data (93% fail rate)
- **SHAP** analysis identified Feature 10 as the dominant predictor — chips with higher Feature 10 values are significantly more likely to pass
- **Clustering** found no discrete failure modes — SECOM failures result from continuous multi-dimensional process drift
- **CNN** achieved 83.1% accuracy classifying 8 wafer map failure types — Edge-Ring F1 0.95, Center F1 0.94

---

## Datasets

- **SECOM** — UCI Machine Learning Repository — 1,567 chips, 590 features, 93% fail rate
- **WM-811K** — Kaggle — 811,457 wafer maps, 8 labeled failure types

---

## Tech Stack

```
Python 3.9 | scikit-learn | XGBoost | LightGBM | SHAP
TensorFlow/Keras | imbalanced-learn | Streamlit | Pandas | NumPy
```

---

## Project Structure

```
SemiSight/
├── data/
│   └── processed/     ← cleaned and processed data files
├── notebooks/         ← Jupyter notebooks for each phase
├── models/            ← saved model files
├── dashboard/         ← Streamlit app
└── reports/           ← saved plots and figures
```

---

## Author

Ahmed Zabir Hussain — Physics | Data Science
