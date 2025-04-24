import requests
import json

url = 'http://127.0.0.1:8000/predict'

# Correct data payload for Iris prediction
iris_data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

headers = {'Content-type': 'application/json'}  # Important to set the content type
response = requests.post(url, data=json.dumps(iris_data), headers=headers)

print("Response Status Code:", response.status_code)
print("Response Content:", response.json())