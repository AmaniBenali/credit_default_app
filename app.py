import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import helpers  # make sure helpers.py is in the same folder!

st.title("Credit Card Default Prediction")

# Load models once
@st.cache_resource
def load_models():
    models = {
        'Logistic Regression': joblib.load("models/logreg.pkl"),
        'KNN': joblib.load("models/knn.pkl"),
        'Decision Tree': joblib.load("models/dtree.pkl"),
        'SVM': joblib.load("models/svm.pkl"),
        'XGBoost': joblib.load("models/xgb.pkl"),

    }
    scaler = joblib.load("models/scaler.pkl")
    return models, scaler

models, scaler = load_models()

# Upload input CSV or use example
uploaded_file = st.file_uploader("Upload input CSV (same features as training data)", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
else:
    input_df = pd.read_csv("models/example_input.csv")
    st.info("Using example input data. Upload your CSV to use your own data.")

st.write("Input data preview:")
st.dataframe(input_df.head())

# Preprocess input (scale)
input_ids = input_df['ID']
input_features = input_df.drop('ID', axis=1)
input_scaled = scaler.transform(input_features)

st.markdown("---")
st.header("Model Predictions")

results = {}

for name, model in models.items():
    preds = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[:,1] if hasattr(model, "predict_proba") else None
    results[name] = (preds, proba)
    st.subheader(name)
    st.write("Predictions:")
    st.write(preds)
    if proba is not None:
        st.write("Predicted probabilities for default:")
        st.write(proba.round(3))

# Optional: Upload true labels to show metrics
true_labels_file = st.file_uploader("Upload true labels CSV (with 'def_pay' column) to evaluate models", type=["csv"])
if true_labels_file:
    true_df = pd.read_csv(true_labels_file)
    y_true = true_df['def_pay']
    st.header("Model Performance Metrics")
    for name, (preds, proba) in results.items():
        st.subheader(name)
        st.write(classification_report(y_true, preds, zero_division=0))
        cm = confusion_matrix(y_true, preds)
        st.write("Confusion Matrix:")
        st.write(cm)
        acc = accuracy_score(y_true, preds)
        st.write(f"Accuracy: {acc:.3f}")
        if proba is not None:
            fpr, tpr, _ = roc_curve(y_true, proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], linestyle='--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {name}')
            ax.legend(loc='lower right')
            st.pyplot(fig)
