from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()
model = joblib.load("model.joblib")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Hello! Welcome to the IRIS classifier."}

@app.post("/predict")
def predict(data: IrisInput):
    x = pd.DataFrame([data.dict()])
    pred = model.predict(x)
    return {"Predicted Sspecies": pred[0]}
