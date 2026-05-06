import streamlit as st
import numpy as np
import joblib

st.title("CTG Fetal Health Predictor")

model = joblib.load("rf_best_model.pkl")
scaler = joblib.load("scaler.pkl")

feature_names = [
    'b', 'e', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'DR', 'LB',
    'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max',
    'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance',
    'Tendency', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD',
    'FS', 'SUSP', 'CLASS'
]

inputs = []

for col in feature_names:
    val = st.number_input(col, value=0.0)
    inputs.append(val)

input_data = np.array([inputs])

if st.button("Predict"):
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.success("Normal")
        elif prediction == 2:
            st.warning("Suspect")
        else:
            st.error("Pathologic")

    except Exception as e:
        st.error(f"Error: {e}")
