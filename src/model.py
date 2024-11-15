import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

import pickle
from pathlib import Path
import pickle

dataset_mappers = {
    "california_housing": {
        "dataset": fetch_california_housing,
        "target_column": "price"
    },
    "diabetes": {
        "dataset": load_diabetes,
        "target_column": "diabetes"
    }
}

def get_data(dataset_type: str) -> pd.DataFrame:
    data = dataset_mappers[dataset_type]["dataset"]()

    transformed_data = pd.DataFrame(data["data"], columns=data["feature_names"])
    transformed_data[dataset_mappers[dataset_type]["target_column"]] = data["target"]

    return transformed_data

def get_train_test(data: pd.DataFrame, target_feature: str, test_size: float = 0.3, random_state: int = 42) -> list[pd.DataFrame]:
    X = data.drop(columns=[target_feature])
    y = data[target_feature]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_linear(x: pd.Series, y: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(x, y)

    return model

def train_ridge(x: pd.Series, y: pd.Series) -> Ridge:
    model = Ridge()
    model.fit(x, y)

    return model

def test_accuracy(pred_data: np.array, true_data: np.array) -> float:
    r2 = r2_score(true_data, pred_data) 
    mse = mean_squared_error(true_data, pred_data)

    return r2 * 100, mse

def get_model(model_type: str, data: pd.DataFrame, target_feature: str) -> LinearRegression:
    X_train, X_test, y_train, y_test = get_train_test(data=data, target_feature=target_feature, test_size=0.3, random_state=42)

    model = train_linear(x=X_train, y=y_train) if model_type == "linear" else train_ridge(x=X_train, y=y_train)

    r_squared, mse = test_accuracy(model.predict(X_test), y_test)

    print(f'R^2 for {model_type} is {r_squared}, mse - {mse}')

    return model

def get_prediction(model_type: str, dataset_type: str, x: float) -> float:
    model_path = Path(__file__).parent.parent / f'models/{model_type}_{dataset_type}.pkl'
    if model_path.exists():
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    else:
        model = get_model(model_type=model_type, data=get_data(dataset_type=dataset_type), target_feature=dataset_mappers[dataset_type]["target_column"])
        with open(model_path, "wb") as file:
            pickle.dump(model, file, protocol=5)
        
    prediction = model.predict([x])

    return prediction[0]

# get_prediction(model_type="ridge", dataset_type="diabetes", x=[0.038, 0.05, 0.0616, 0.021, -0.0442, -0.0348, -0.0434, -0.0025, 0.0199, -0.0176])
#
