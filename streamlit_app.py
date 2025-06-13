import streamlit as st
import numpy as np
import joblib

# Loading the pickle
model_file = joblib.load("random_forest_model.pkl")

if isinstance(model_file, tuple):
    model, scaler, features = model_file
elif isinstance(model_file, dict):
    model = model_file['model']
    scaler = model_file.get('scaler', None)
    features = model_file.get('features', [])
else:
    model = model_file
    scaler = None
    features = []

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("🫀 Heart Disease Prediction Tool")
st.markdown("### Enter Patient Details Below:")

with st.form("input_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", value=120)
    chol = st.number_input("Cholesterol (mg/dl)", value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120?", [0, 1])
    restecg = st.selectbox("ECG Results", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", value=160)
    exang = st.selectbox("Exercise-induced Angina?", [0, 1])
    oldpeak = st.number_input("ST Depression", value=1.0)
    slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3])

    submit = st.form_submit_button("Predict")

if submit:
    features_vals = np.array([[age, sex, cp, trestbps, chol, fbs,restecg, thalach, exang, oldpeak,slope, ca, thal]])

    if scaler is not None:
        features_vals = scaler.transform(features_vals)

    try:
        prediction = model.predict(features_vals)[0]

        if prediction == 1:
            st.error("⚠ Risk of Heart Disease detected.")
        else:
            st.success("✅ No Heart Disease detected.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
