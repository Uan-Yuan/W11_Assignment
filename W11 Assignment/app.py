from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import xgboost as xgb
import os

app = FastAPI()

# ===== Load the trained model =====
model_path = "model/model.json"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. "
                            f"Run train_model.py first.")

booster = xgb.Booster()
booster.load_model(model_path)

FEATURE_NAMES = [f"lag_{i}" for i in range(1, 7)]

class SalesInput(BaseModel):
    m1: float
    m2: float
    m3: float
    m4: float
    m5: float
    m6: float

# ===== Serve frontend =====
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def read_root():
    # Serve the HTML file
    return FileResponse("frontend/index.html")

# ===== Prediction endpoint =====
@app.post("/predict")
def predict_sales(data: SalesInput):
    # Order matters: must match lag_1 ... lag_6
    x = np.array([[data.m1, data.m2, data.m3, data.m4, data.m5, data.m6]])
    dmatrix = xgb.DMatrix(x, feature_names=FEATURE_NAMES)
    pred = booster.predict(dmatrix)[0]
    return {"prediction": float(pred)}
