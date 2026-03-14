import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import os
import tensorflow as tf

st.set_page_config(page_title="SemiSight", page_icon="🔬", layout="wide")

BASE_PATH = os.path.expanduser('~/Documents/SemiSight')

if 'loaded' not in st.session_state:
    st.session_state.rf_model = joblib.load(f'{BASE_PATH}/models/champion_model.pkl')
    st.session_state.cnn_model = tf.keras.models.load_model(f'{BASE_PATH}/models/cnn_wafer_map.keras')
    st.session_state.X_test = np.load(f'{BASE_PATH}/data/processed/X_test_selected.npy')
    st.session_state.y_test = np.load(f'{BASE_PATH}/data/processed/y_test.npy')
    st.session_state.wafer_maps = np.load(f'{BASE_PATH}/data/processed/wafer_maps.npy')
    st.session_state.wafer_labels = np.load(f'{BASE_PATH}/data/processed/wafer_labels.npy', allow_pickle=True)
    st.session_state.loaded = True

rf_model = st.session_state.rf_model
cnn_model = st.session_state.cnn_model
X_test = st.session_state.X_test
y_test = st.session_state.y_test
wafer_maps = st.session_state.wafer_maps
wafer_labels = st.session_state.wafer_labels

CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch']

# Sidebar
st.sidebar.title("🔳 SemiSight")
st.sidebar.markdown("Semiconductor Fault Prediction Platform")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** UCI SECOM + WM-811K")
st.sidebar.markdown("**Models:** Random Forest + CNN")
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
    st.image(f'{BASE_PATH}/reports/model_comparison.png')

    st.subheader("Feature Importance")
    st.image(f'{BASE_PATH}/reports/feature_importance.png')

    st.subheader("SHAP Summary Plot")
    st.image(f'{BASE_PATH}/reports/shap_summary_plot.png')

    st.subheader("SHAP Force Plot — Chip 304 (True Positive)")
    st.image(f'{BASE_PATH}/reports/shap_force_chip304.png')

# PAGE 2
elif page == "Failure Mode Explorer":
    st.title("Failure Mode Explorer")
    st.markdown("""
    This page explores whether the 293 failed chips in the SECOM test set fall into distinct 
    failure mode groups. KMeans clustering was applied across k=2 to k=10, with outlier removal 
    using Local Outlier Factor before clustering.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Failed Chips", "293")
    col2.metric("Outliers Removed", "15")
    col3.metric("Clean Chips", "278")
    col4.metric("Best Silhouette", "0.07")

    st.markdown("**Key Findings:**")
    st.markdown("""
    - 15 outlier chips removed — likely sensor errors or recording anomalies
    - Silhouette scores below 0.07 across all values of k — no meaningful cluster structure found
    - PCA projection explains only 16.3% of variance in 2D — failures spread across 100 dimensions
    - Conclusion: failures result from continuous process drift, not discrete fault types
    """)

    st.subheader("Optimal K Search")
    st.image(f'{BASE_PATH}/reports/clustering_optimal_k.png')

    st.subheader("PCA Visualization — Failed Chips")
    st.image(f'{BASE_PATH}/reports/failure_mode_pca.png')

    st.subheader("Silhouette Scores by K")
    silhouette_data = {
        'K': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Silhouette Score': [0.0632, 0.0700, 0.0441, 0.0361, 0.0450, 0.0430, 0.0420, 0.0444, 0.0466]
    }
    st.dataframe(pd.DataFrame(silhouette_data), use_container_width=True)

# PAGE 3
elif page == "Wafer Map Classifier":
    st.title("Wafer Map CNN Classifier")
    st.markdown("""
    This page uses a CNN trained on 25,519 labeled wafers from the WM-811K dataset to classify 
    8 distinct failure patterns in real time. Select a wafer index to see the wafer map and 
    the CNN's prediction.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Wafers", "811,457")
    col2.metric("Labeled Wafers", "25,519")
    col3.metric("Failure Types", "8")
    col4.metric("Test Accuracy", "83.1%")

    st.markdown("**Key Findings:**")
    st.markdown("""
    - Edge-Ring and Center patterns achieved F1 scores of 0.95 and 0.94
    - Scratch scored only 0.32 F1 — only 238 training samples
    - Near-full achieved perfect recall but low precision due to high class weight
    - Training completed in 11 epochs on Apple M4 GPU
    """)

    st.subheader("Live Wafer Map Prediction")
    wafer_idx = st.slider("Select Wafer Index", 0, len(wafer_maps)-1, 0)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Wafer Map")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(wafer_maps[wafer_idx], cmap='coolwarm', interpolation='nearest')
        ax.set_title(f"True Label: {wafer_labels[wafer_idx]}")
        ax.axis('off')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("CNN Prediction")
        processed = wafer_maps[wafer_idx].reshape(1, 32, 32, 1)
        probs = cnn_model.predict(processed, verbose=0)[0]
        pred_class = CLASSES[np.argmax(probs)]

        st.metric("Predicted Failure Type", pred_class)
        st.metric("Confidence", f"{np.max(probs)*100:.1f}%")
        st.metric("True Label", wafer_labels[wafer_idx])

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(CLASSES, probs, color='steelblue')
        ax.set_xlabel('Probability')
        ax.set_title('Class Probabilities')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Training History")
    st.image(f'{BASE_PATH}/reports/cnn_training_history.png')

    st.subheader("Confusion Matrix")
    st.image(f'{BASE_PATH}/reports/cnn_confusion_matrix.png')

    st.subheader("Per Class F1 Scores")
    results = {
        'Failure Type': ['Edge-Ring', 'Center', 'Donut', 'Random', 'Edge-Loc', 'Loc', 'Near-full', 'Scratch'],
        'F1 Score': [0.95, 0.94, 0.85, 0.79, 0.76, 0.68, 0.62, 0.32],
        'Support': [1936, 859, 111, 173, 1038, 719, 30, 238]
    }
    st.dataframe(pd.DataFrame(results), use_container_width=True)