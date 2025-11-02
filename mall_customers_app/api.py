from fastapi import FastAPI
from pydantic import BaseModel
from app_linear import predict_linear
from app_ridge import predict_ridge
from app_lasso import predict_lasso

app = FastAPI(title="Mall Customers Regression API")

class InputData(BaseModel):
    features: list[float]
    model_type: str  # "linear", "ridge", "lasso"

@app.post("/predict")
def predict(data: InputData):
    if data.model_type == "linear":
        prediction = predict_linear(data.features)
    elif data.model_type == "ridge":
        prediction = predict_ridge(data.features)
    elif data.model_type == "lasso":
        prediction = predict_lasso(data.features)
    else:
        return {"error": "Invalid model type"}
    return {"model": data.model_type, "prediction": prediction}
