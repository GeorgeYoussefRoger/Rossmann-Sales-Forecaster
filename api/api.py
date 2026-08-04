from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

from src.features import add_date_features, add_promo_features

pipeline = joblib.load("models/final_pipeline.pkl")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
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
    return {"status": "ok"}

@app.post("/predict")
def predict(data: StoreData):
    df = pd.DataFrame([data.model_dump()])
    df['Date'] = pd.to_datetime(df['Date'])
    df = add_date_features(df)
    df = add_promo_features(df)
    try:
        prediction = np.expm1(pipeline.predict(df)[0])
        return {"prediction": float(prediction)}
    except Exception as e:
        return {"error": str(e)}