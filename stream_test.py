import streamlit as st
import pickle
import numpy as np
import pandas as pd

@st.cache_resource

def load_model():
    with open("diabetes.pkl","rb") as f:
        return pickle.load(f)
        
model = load_model()

st.title("Diabetes Prediction APP")
st.write("Enter Patient Details")
st.sidebar.header("Patient Inputs")

pregnancies = st.sidebar.text_input("pregnancies","0")
glucose = st.sidebar.text_input("glucose","120")
bp = st.sidebar.text_input("Blood Pressure","70")
skin_thickness = st.sidebar.text_input("skin thickness","20")
insulin = st.sidebar.text_input("insulin","79")
bmi = st.sidebar.text_input("BMI","25.0")
dpf = st.sidebar.text_input("Diabetes Pedigree Function","0.5")
age = st.sidebar.text_input("Age","33")

if st.button("Predict Diabetes Risk"):
    try:
        features = np.array([[
            int(pregnancies),
            int(glucose),
            int(bp),
            int(skin_thickness),
            int(insulin),
            float(bmi),
            float(dpf),
            int(age)
        ]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]*100
        
        st.subheader("Result")
        
        if prediction ==1:
            st.error(f"Likely Diabetic \n\n Risk Score : {probability:.1f}%")
        else:
            st.success(f"Likely Not Diabetic \n\n Risk Score : {probability:.1f}%")
        st.progress(float(probability/100))
    except ValueError:
        st.error("Please Enter Valid Numerical Values.")