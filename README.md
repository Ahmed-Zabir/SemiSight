# SemiSight 🔳 
### Semiconductor Yield Analytics Platform

Semiconductor fabs lose millions of dollars per percentage point of yield 
loss. SemiSight is an end-to-end analytics platform that predicts chip 
failure before electrical test, explains the root causes driving yield loss, 
and automatically classifies wafer defect patterns, giving process engineers 
actionable intelligence, faster.

---

## The Problem
- Yield loss in semiconductor manufacturing is costly and often poorly understood
- Process engineers are drowning in high dimensional sensor data with no 
  clear signal
- Wafer defect patterns are classified manually, slowly, and inconsistently

## What SemiSight Does
| Capability | What It Means For You |
|---|---|
| **Yield Prediction** | Predicts chip pass/fail from process sensor data before electrical test |
| **Root Cause Analysis** | SHAP explainability identifies which process parameters are driving failure |
| **Process Drift Detection** | Flags when manufacturing process is drifting before yield degrades |
| **Wafer Defect Classification** | CNN automatically classifies 8 wafer map failure types from binary images |
| **Interactive Dashboard** | All insights accessible in a 3-page Streamlit dashboard — no code required |

---

## Results
- **AUC-ROC 0.768** on 93% imbalanced real-world fab data (SECOM dataset)
- **83.1% wafer defect classification accuracy** across 8 failure types
- **Edge-Ring F1: 0.95 | Center F1: 0.94** — highest-frequency defect types 
  classified with production-grade accuracy
- **Continuous process drift confirmed** — no discrete failure clusters, 
  meaning yield loss is driven by gradual parameter shift, not single-point 
  failures
- Feature 10 identified as dominant yield driver via SHAP — actionable 
  signal for process engineers

---

## Dashboard
3-page interactive Streamlit app covering yield prediction, SHAP 
explainability, and wafer defect classification.

**Run locally:**
```bash
cd ~/Documents/SemiSight
source semisight_env/bin/activate
streamlit run dashboard/app.py
```

---

## Tech Stack
Python 3.9 | scikit-learn | XGBoost | LightGBM | SHAP | 
TensorFlow/Keras | imbalanced-learn | Streamlit | Pandas | NumPy

---

## Project Structure
```
SemiSight/
├── data/processed/     ← cleaned and processed data files
├── notebooks/          ← phase-by-phase analysis notebooks
├── models/             ← saved model files
├── dashboard/          ← Streamlit app
└── reports/            ← plots and figures
```

---

## Datasets
- **SECOM** (UCI ML Repository) — 1,567 chips, 590 process features
- **WM-811K** (Kaggle) — 811,457 labeled wafer maps, 8 failure types

---

## Author
**Ahmed Zabir Hussain** — Physics & Data Science  
Open to consulting engagements in semiconductor yield analytics and 
wafer defect classification.  
