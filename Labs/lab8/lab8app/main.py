from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel
import mlflow
import pandas as pd
app = FastAPI(
    title="ML Flow predictor",
    description="Use a ML Flow Model to create predictions",
    version="0.1",
)

# Defining path operation for root endpoint
@app.get('/')
def main():
	return {'message': 'This uses a pretrained Ml Flow model trained on the Iris Dataset.'}

class request_body(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.on_event('startup')
def load_artifacts():
    global model, feature_names
    model_name = "best_iris_model"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Lab8.1_Iris")
    model_uri = f"models:/{model_name}/latest"
    model = mlflow.pyfunc.load_model(model_uri)
    feature_names = [
        "sepal length",
        "sepal width",
        "petal length",
        "petal width",
    ]

# Defining path operation for /predict endpoint
@app.post('/predict')
def predict(data : request_body):
    input_data = pd.DataFrame([data.dict()])
    input_data = input_data[
            ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        ]
    input_data.columns = [  # Rename columns to match training data
            "sepal length",
            "sepal width",
            "petal length",
            "petal width",
        ]
    predictions = model.predict(input_data)
    return {'Predictions': predictions.tolist()}