from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load('claim_predictor_model.pkl')

app = FastAPI()

# Define input data structure
class ClaimData(BaseModel):
    age: float
    bmi: float
    bloodpressure: float
    children: int
    smoker: str
    region: str

# Endpoint to receive user input and return the prediction
@app.post('/predict')
def predict(data: ClaimData):
    input_data = np.array([[
        data.age,
        data.bmi,
        data.bloodpressure,
        data.children,
        1 if data.smoker == 'Yes' else 0,
        data.region
    ]])

    # Get the prediction
    prediction = model.predict(input_data)

    return {'prediction': prediction[0]}