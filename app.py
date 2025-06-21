import streamlit as st
import pandas as pd
import joblib
import os

# Charger les modèles et le scaler
scaler = joblib.load("models/scaler (2).pkl")
models = {
    "Logistic Regression": joblib.load("models/logreg.pkl"),
    "K-Nearest Neighbors (K=13)": joblib.load("models/knn.pkl"),
    "Decision Tree": joblib.load("models/dtree.pkl"),
    "SVM": joblib.load("models/svm.pkl"),
    "XGBoost": joblib.load("models/xgb.pkl"),
}

# Liste exacte des colonnes utilisées pour l'entraînement (sans 'def_pay')
expected_columns = [
    'ID', 'LIMIT_BAL', 'AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'SEX_female', 'MARRIAGE_married', 'MARRIAGE_single',
    'EDUCATION_graduate_school', 'EDUCATION_university', 'EDUCATION_high_school', 'EDUCATION_others'
]

# Interface Streamlit
st.title("Credit Card Default Prediction")
st.markdown("**Upload your processed .xlsx file (with the same preprocessing as training)**")

uploaded_file = st.file_uploader("Upload input file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        st.subheader("Input Data Preview:")
        st.write(df.head())

        # Supprimer 'def_pay' si elle est présente (erreur fréquente)
        if 'def_pay' in df.columns:
            st.warning("Column 'def_pay' detected and removed automatically.")
            df = df.drop(columns=['def_pay'])

        # Vérification des colonnes
        missing_cols = set(expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_columns)

        if missing_cols:
            st.error(f"Missing columns in uploaded file: {missing_cols}")
        elif extra_cols:
            st.warning(f"Extra columns found and ignored: {extra_cols}")
            df = df[expected_columns]  # Réordonner et garder uniquement les colonnes nécessaires
        else:
            df = df[expected_columns]

            # Sauvegarder l’ID pour affichage
            if 'ID' in df.columns:
                client_ids = df['ID'].values
                df = df.drop(columns=['ID'])

            # Standardisation
            input_scaled = scaler.transform(df)

            st.success("File successfully processed. Make a prediction:")

            model_choice = st.selectbox("Choose a model", list(models.keys()))
            if st.button("Predict"):
                model = models[model_choice]
                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)[:, 1]

                result_df = pd.DataFrame({
                    "Client ID": client_ids,
                    "Prediction (0=No Default, 1=Default)": prediction,
                    "Probability of Default": prediction_proba
                })

                st.subheader("Prediction Results:")
                st.write(result_df)

                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"An error occurred while processing your file: {str(e)}")
