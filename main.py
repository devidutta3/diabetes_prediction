from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Create FastAPI App
app = FastAPI(
    title="Diabetes Prediction API",
    description="Predict Diabetes using Logistic Regression",
    version="1.0"
)

# Load Model and Scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")


# Input Schema
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


# Home Route
@app.get("/")
def home():
    return {
        "message": "Diabetes Prediction API Running 🚀"
    }


# Health Check Route
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Prediction Route
@app.post("/predict")
def predict(data: PatientData):

    features = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])

    # Scale Features
    scaled_features = scaler.transform(features)

    # Predict
    prediction = model.predict(scaled_features)[0]

    # Probability
    probability = model.predict_proba(scaled_features)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "result": "Diabetic" if prediction == 1 else "Non-Diabetic"
    }
