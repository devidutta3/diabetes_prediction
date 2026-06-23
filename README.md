# Diabetes Prediction API using Logistic Regression
<img src="assests\ChatGPT Image Jun 23, 2026, 11_33_52 PM.png">

<h1>OverView</h1> 

This project is a Machine Learning-based Diabetes Prediction System built using Logistic Regression and deployed using FastAPI.

The model predicts whether a patient is diabetic or non-diabetic based on various medical parameters such as Glucose Level, Blood Pressure, BMI, Insulin, Age, and more.

The API accepts patient information as input and returns:

* Prediction (Diabetic / Non-Diabetic)
* Prediction Probability
* Health Status Response



## Features

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Missing Value Handling using Median Imputation
* Feature Scaling using StandardScaler
* Logistic Regression Model Training
* Model Persistence using Joblib
* FastAPI REST API
* Interactive Swagger Documentation

---

## Dataset

Dataset Used: Pima Indians Diabetes Dataset

### Features

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age

### Target

* 0 → Non-Diabetic
* 1 → Diabetic

---

## Project Workflow

Dataset
→ Data Cleaning
→ Feature Engineering
→ Train-Test Split
→ Feature Scaling
→ Logistic Regression
→ Model Evaluation
→ Save Model
→ FastAPI Deployment

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* FastAPI
* Uvicorn
* Joblib
* Pydantic

---

## Model Performance

Algorithm Used:

* Logistic Regression

Accuracy Achieved:

* 75.32%

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd diabetes-prediction-api
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Model Training

```bash
python train.py
```

Generated files:

```text
model.pkl
scaler.pkl
```

---

## Run FastAPI Server

```bash
python -m uvicorn main:app --reload
```

Server URL:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Home Endpoint

```http
GET /
```

Response:

```json
{
  "message": "Diabetes Prediction API Running 🚀"
}
```

---

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

---

### Prediction Endpoint

```http
POST /predict
```

Request Body:

```json
{
  "Pregnancies": 6,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 168,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

Response:

```json
{
  "prediction": 1,
  "probability": 0.82,
  "result": "Diabetic"
}
```

---

## Project Structure

```text
diabetes-prediction-api/
│
├── data/
│   ├── diabetes.csv
│   └── clean_dataset.csv
│
├── feature.py
├── train.py
├── main.py
│
├── model.pkl
├── scaler.pkl
│
├── requirements.txt
└── README.md
```

---

## Future Improvements

* Add Frontend Dashboard
* Deploy on Render
* Deploy on Railway
* Docker Containerization
* Cloud Deployment
* Model Comparison (Random Forest, XGBoost)
* Advanced Evaluation Metrics

---

## Author

Krishna

B.Tech AI & ML Student

Passionate about Machine Learning, FastAPI, and AI-powered solutions.
