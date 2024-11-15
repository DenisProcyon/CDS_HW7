from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.model import get_prediction

app = FastAPI()

class PredictionRequest(BaseModel):
    mod_type: str
    dataset_type: str
    x: List[float]

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        prediction = get_prediction(
            model_type=request.mod_type, 
            dataset_type=request.dataset_type, 
            x=request.x
        )

        return {"prediction": prediction}
    except Exception as e:
        print(f'Error for parameters - {request.dict()} - {e}')
        raise HTTPException(502, detail=str(e))
