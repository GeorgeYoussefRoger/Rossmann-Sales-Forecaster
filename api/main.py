from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

try:
    model = joblib.load('models/final_model.pkl')
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

class StoreData(BaseModel):
    Store: int
    DayOfWeek: int
    Date: str
    Promo: int
    StateHoliday: str
    SchoolHoliday: int
    StoreType: str
    Assortment: str
    CompetitionDistance: float
    CompetitionOpenSinceMonth: int
    CompetitionOpenSinceYear: int
    Promo2: int
    Promo2SinceWeek: int
    Promo2SinceYear: int
    PromoInterval: str
    sales_lag_7: float
    sales_lag_14: float
    rolling_mean_7: float
    rolling_mean_30: float

@app.get("/")
def root():
    return {"message": "Welcome to the Rossmann Sales Prediction API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: StoreData):
    df = pd.DataFrame([data.model_dump()])
    df['Date'] = pd.to_datetime(df['Date'])
    try:
        prediction = np.expm1(model.predict(df)[0])
        return {"predicted_sales": float(prediction)}
    except Exception as e:
        return {"error": str(e)}