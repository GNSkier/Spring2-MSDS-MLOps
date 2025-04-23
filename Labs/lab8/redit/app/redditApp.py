from fastapi import FastAPI
import uvicorn

app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)

# Defining path operation for root endpoint
@app.get('/')
def main():
	return {'message': 'This is a model for classifying Reddit comments'}

# Defining path operation for /name endpoint
@app.get('/{name}')
def hello_name(name : str):
	return {'message': f'Hello {name}'}