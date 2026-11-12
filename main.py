from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

model = joblib.load("diabetes.pkl")

app = FastAPI(title="Diabetes Prediction")


class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


@app.post("/predict")
def predict(data: DiabetesInput):

    input_array = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])

    prediction = model.predict(input_array)[0]

    return {
        "prediction": int(prediction),
        "result": "diabetes" if prediction == 1 else "not diabetes"
    }