import streamlit as st
import numpy as np
import joblib

# Load the trained model
model, scaler, features = joblib.load("random_forest_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀", layout="centered")
st.title("🫀 Heart Disease Prediction Tool")
st.write("Enter the following health parameters to predict the risk of heart disease.")

feature_descriptions = {
    "age": "Age (years)",
    "sex": "Sex (0 = Female, 1 = Male)",
    "cp": "Chest Pain Type (0 = Typical, 1 = Atypical, 2 = Non-anginal, 3 = Asymptomatic)",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120? (1 = Yes, 0 = No)",
    "restecg": "ECG Results (0 = Normal, 1 = Abnormal, 2 = LV Hypertrophy)",
    "thalach": "Max Heart Rate Achieved",
    "exang": "Exercise-induced Angina (1 = Yes, 0 = No)",
    "oldpeak": "ST Depression",
    "slope": "Slope of Peak Exercise ST Segment (0 = Upsloping, 1 = Flat, 2 = Downsloping)",
    "ca": "Number of Major Vessels Colored (0–3)",
    "thal": "Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)"
}

input_data = []

for feat in features:
    desc = feature_descriptions.get(feat, feat)
    val = st.number_input(f"{desc}:", value=0.0, format="%.2f")
    input_data.append(val)

if st.button("Predict"):
    try:
        data_scaled = scaler.transform([input_data])
        prediction = model.predict(data_scaled)[0]
        if prediction == 1:
            st.error("⚠️ High risk of heart disease detected!")
        else:
            st.success("✅ No heart disease detected.")
    except Exception as e:
        st.error(f"Error: {e}")
