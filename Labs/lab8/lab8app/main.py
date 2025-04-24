from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel
import mlflow

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
    iris_parameters : str

@app.on_event('startup')
def load_artifacts():
    global model
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Lab6_Iris")
    model_uri = f"models:/{model_name}/latest"
    model = mlflow.pyfunc.load_model(model_uri)


# Defining path operation for /predict endpoint
@app.post('/predict')
def predict(data : request_body):
    X = [data.reddit_comment]
    predictions = model_pipeline.predict_proba(X)
    return {'Predictions': predictions}