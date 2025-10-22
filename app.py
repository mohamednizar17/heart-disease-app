import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model (update path if running locally)
@st.cache_resource
def load_model():
    return joblib.load('full_pipeline.pkl')  # Adjust for local

model = load_model()

st.title("Heart Disease Risk Predictor")
st.write("Enter patient details to predict heart disease probability.")

# Input form (based on dataset features)
age = st.slider("Age", 29, 77, 50)
sex = st.selectbox("Sex (0=Male, 1=Female)", [0, 1])
cp = st.selectbox("Chest Pain Type (cp: 0-3)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure (trestbps)", 94, 200, 120)
chol = st.slider("Serum Cholestoral (chol)", 126, 564, 200)
fbs = st.selectbox("Fasting Blood Sugar >120 (fbs: 0/1)", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.slider("Max Heart Rate (thalach)", 71, 202, 150)
exang = st.selectbox("Exercise Induced Angina (0/1)", [0, 1])
oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0)
slope = st.selectbox("ST Segment Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Major Vessels (ca: 0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal: 1-3)", [1, 2, 3])

if st.button("Predict Risk"):
    # Prepare input
    input_data = pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol],
        'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach], 'exang': [exang],
        'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
    })
    
    # Predict
    prob = model.predict_proba(input_data)[0, 1]
    prediction = "High Risk (Disease Likely)" if prob > 0.5 else "Low Risk"
    
    st.success(f"Prediction: {prediction}")
    st.write(f"Heart Disease Probability: {prob:.2%}")
