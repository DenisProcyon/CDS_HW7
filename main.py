import requests
import json

import random

model_types = ["linear", "ridge"]
dataset_types = ["diabetes", "california_housing"]
x_len = {
    "diabetes": 10,
    "california_housing": 8
}

def get_random_x(n: int) -> list[int]:
    return [random.uniform(0, 1000) for _ in range(n)]

def main():
    model_type = model_types[random.randint(0, len(model_types) - 1)]
    dataset_type = dataset_types[random.randint(0, len(dataset_types) - 1)]

    x = get_random_x(x_len[dataset_type])

    model_request = {
        "mod_type": model_type,
        "dataset_type": dataset_type,
        "x": x
    }

    url = "http://127.0.0.1:8000/predict" 

    model_response = requests.post(url, data=json.dumps(model_request))

    if model_response.status_code == 200:
        print(f'Prediction for {dataset_type} using {model_type} regression is {model_response.json()["prediction"]}')
    if model_response.status_code == 502:
        print(f'Error on the side of server: {model_response.content}')
    if model_response.status_code == 422:
        print(f'Bad parameters: {model_response.content}')
        

if __name__ == "__main__":
    main()