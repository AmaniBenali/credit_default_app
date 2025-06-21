import streamlit as st
import pandas as pd
import joblib
import os

# Load models and scaler
models_dir = "models/"
scaler = joblib.load(models_dir + "scaler.pkl")
models = {
    "Logistic Regression": joblib.load(models_dir + "logreg.pkl"),
    "KNN": joblib.load(models_dir + "knn.pkl"),
    "Decision Tree": joblib.load(models_dir + "dtree.pkl"),
    "SVM": joblib.load(models_dir + "svm.pkl"),
    "XGBoost": joblib.load(models_dir + "xgb.pkl"),
}

st.title("Credit Card Default Prediction")
st.write("Upload your processed .xls file (with the exact same preprocessing used for training).")

uploaded_file = st.file_uploader("Upload input file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        input_df = pd.read_excel(uploaded_file)

        st.subheader("Input Data Preview:")
        st.dataframe(input_df.head())

        # Drop 'ID' column if exists
        if 'ID' in input_df.columns:
            input_features = input_df.drop(columns=['ID'])
        else:
            input_features = input_df.copy()

        # Scale features
        input_scaled = scaler.transform(input_features)

        st.subheader("Predictions:")
        for name, model in models.items():
            pred = model.predict(input_scaled)
            prob = model.predict_proba(input_scaled)[:, 1]  # Probability of default

            st.markdown(f"**{name}**")
            st.write(f"Prediction: {'Default' if pred[0] == 1 else 'No Default'}")
            st.write(f"Probability of Default: {prob[0]:.2%}")
            st.markdown("---")

    except Exception as e:
        st.error("An error occurred while processing your file. Make sure it's the correctly processed version with expected columns.")
        st.exception(e)
else:
    st.info("Please upload a .xlsx file containing the processed input data.")
