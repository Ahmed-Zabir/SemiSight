import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="SemiSight", page_icon="🔬", layout="wide")

BASE_PATH = os.path.expanduser('~/Documents/SemiSight')

st.sidebar.title("🔬 SemiSight")
st.sidebar.markdown("Semiconductor Fault Prediction Platform")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** UCI SECOM + WM-811K")
st.sidebar.markdown("**Models:** Random Forest + CNN")
st.sidebar.markdown("**Phases:** EDA → Preprocessing → Modeling → SHAP → Clustering → CNN")

page = st.sidebar.radio("Navigate", [
    "SECOM Predictor",
    "Failure Mode Explorer",
    "Wafer Map Classifier"
])

# ── PAGE 1 ──
if page == "SECOM Predictor":
    st.title("SECOM Chip Pass/Fail Predictor")

    st.markdown("""
    This page shows the results of a Random Forest classifier trained on the UCI SECOM semiconductor 
    manufacturing dataset. The goal is to predict whether a chip will pass or fail quality testing 
    based on 100 sensor readings collected during the manufacturing process.
    """)

    st.markdown("**Dataset Info:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Chips", "1,567")
    col2.metric("Features", "100")
    col3.metric("Pass Rate", "6.6%")
    col4.metric("AUC-ROC", "0.768")

    st.markdown("**Key Findings:**")
    st.markdown("""
    - Severe class imbalance — 93.4% of chips fail, making accuracy a misleading metric
    - SMOTE oversampling was applied to the training set to balance classes
    - Default Random Forest outperformed tuned models, XGBoost, and LightGBM on Pass F1 score
    - Feature 10 was the single most important predictor globally across all SHAP analyses
    - Pass chips tend to have higher Feature 10 values — a potential real manufacturing threshold
    """)

    st.subheader("Model Comparison")
    st.markdown("Comparison of all models evaluated. F1 score for the Pass class was the primary metric due to class imbalance.")
    st.image(f'{BASE_PATH}/reports/model_comparison.png')

    st.subheader("Feature Importance")
    st.markdown("Top features ranked by importance in the champion Random Forest model. Feature 10 dominates as the strongest predictor of chip outcome.")
    st.image(f'{BASE_PATH}/reports/feature_importance.png')

    st.subheader("SHAP Summary Plot")
    st.markdown("SHAP values show how each feature pushes predictions toward Pass or Fail. Red dots indicate high feature values, blue indicate low. Features higher on the chart have greater overall impact.")
    st.image(f'{BASE_PATH}/reports/shap_summary_plot.png')

    st.subheader("SHAP Force Plot — Chip 304 (True Positive)")
    st.markdown("This chip was correctly predicted as Pass. Feature 10 was the strongest advocate pushing toward Pass, while Feature 72 was the biggest dissenter pushing toward Fail.")
    st.image(f'{BASE_PATH}/reports/shap_force_chip304.png')

# ── PAGE 2 ──
elif page == "Failure Mode Explorer":
    st.title("Failure Mode Explorer")

    st.markdown("""
    This page explores whether the 293 failed chips in the SECOM test set fall into distinct 
    failure mode groups. KMeans clustering was applied across k=2 to k=10, with outlier removal 
    using Local Outlier Factor (LOF) before clustering.
    """)

    st.markdown("**Dataset Info:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Failed Chips", "293")
    col2.metric("Outliers Removed", "15")
    col3.metric("Clean Chips", "278")
    col4.metric("Best Silhouette", "0.07")

    st.markdown("**Key Findings:**")
    st.markdown("""
    - 15 outlier chips were identified and removed — likely sensor errors or recording anomalies
    - Silhouette scores below 0.07 across all values of k — no meaningful cluster structure found
    - PCA projection explains only 16.3% of variance in 2D — failures are spread across 100 dimensions
    - Conclusion: failures result from continuous multi-dimensional process drift, not discrete fault types
    - This is a valid and informative finding — it tells engineers there is no single dominant failure mode to target
    """)

    st.subheader("Optimal K Search")
    st.markdown("Inertia and silhouette scores across k=2 to k=10. The silhouette scores remain flat and near zero throughout — confirming no natural cluster structure exists in the data.")
    st.image(f'{BASE_PATH}/reports/clustering_optimal_k.png')

    st.subheader("PCA Visualization — Failed Chips")
    st.markdown("278 failed chips projected to 2D using PCA. The failures form one continuous cloud with no separable boundaries, confirming the clustering findings visually.")
    st.image(f'{BASE_PATH}/reports/failure_mode_pca.png')

    st.subheader("Silhouette Scores by K")
    silhouette_data = {
        'K': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Silhouette Score': [0.0632, 0.0700, 0.0441, 0.0361, 0.0450, 0.0430, 0.0420, 0.0444, 0.0466]
    }
    st.dataframe(pd.DataFrame(silhouette_data), use_container_width=True)

# ── PAGE 3 ──
elif page == "Wafer Map Classifier":
    st.title("Wafer Map CNN Classifier")

    st.markdown("""
    This page shows the results of a Convolutional Neural Network trained on the WM-811K wafer map 
    dataset to classify 8 distinct failure patterns. Each wafer map is a 2D grid where failed dies 
    appear in red and passing dies in white. The CNN learns to recognize spatial failure patterns 
    the same way it would recognize objects in photos.
    """)

    st.markdown("**Dataset Info:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Wafers", "811,457")
    col2.metric("Labeled Wafers", "25,519")
    col3.metric("Failure Types", "8")
    col4.metric("Test Accuracy", "83.1%")

    st.markdown("**Key Findings:**")
    st.markdown("""
    - Edge-Ring and Center patterns achieved F1 scores of 0.95 and 0.94 — visually distinct and easy to learn
    - Scratch pattern scored only 0.32 F1 — only 238 training samples, not enough to learn the diagonal line pattern
    - Near-full achieved perfect recall (1.00) but low precision (0.45) due to high class weight amplification
    - Training completed in 11 epochs with early stopping — 2-3 seconds per epoch on MacBook
    - Model has 315,272 parameters — lightweight enough for local deployment
    """)

    st.subheader("Failure Type Examples")
    st.markdown("One example wafer map per failure type. Each pattern has a distinct spatial signature — from the diagonal lines of Scratch to the complete red coverage of Near-full.")
    st.image(f'{BASE_PATH}/reports/wafer_map_examples.png')

    st.subheader("Training History")
    st.markdown("Model accuracy and loss across 11 epochs. Both training and validation metrics improve together with no significant gap — indicating the model generalizes well without overfitting.")
    st.image(f'{BASE_PATH}/reports/cnn_training_history.png')

    st.subheader("Confusion Matrix")
    st.markdown("CNN predictions vs true labels on 5,104 test wafers. The bright diagonal confirms strong performance across most classes. Off-diagonal values show where the model confuses similar patterns — mainly between Loc and Edge-Loc.")
    st.image(f'{BASE_PATH}/reports/cnn_confusion_matrix.png')

    st.subheader("Per Class F1 Scores")
    results = {
        'Failure Type': ['Edge-Ring', 'Center', 'Donut', 'Random', 'Edge-Loc', 'Loc', 'Near-full', 'Scratch'],
        'F1 Score': [0.95, 0.94, 0.85, 0.79, 0.76, 0.68, 0.62, 0.32],
        'Support': [1936, 859, 111, 173, 1038, 719, 30, 238]
    }
    st.dataframe(pd.DataFrame(results), use_container_width=True)