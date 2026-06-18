from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import pandas as pd
from pathlib import Path


# =====================
# LOAD MODEL
# =====================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR /
    "models" /
    "xgboost.pkl"
)

model = joblib.load(
    MODEL_PATH
)


# =====================
# CREATE API
# =====================

app = FastAPI(
    title="Crypto Pump Detector",
    description="Predict crypto pump probability using XGBoost",
    version="1.0"
)


# =====================
# INPUT SCHEMA
# =====================

class PumpFeatures(BaseModel):

    price_change_1: float
    price_change_3: float
    price_change_6: float

    volume_change_1: float
    volume_change_3: float
    volume_change_6: float

    distance_ma_long: float

    volatility_short: float
    volatility_long: float

    momentum_10: float

    volume_zscore: float
    volume_spike: int

    bollinger_width: float
    bollinger_breakout: int

    rsi: float

    relative_volume: float
    breakout_strength: float
    green_candle_count: float

    volume_acceleration: float
    rsi_change: float

    above_ma_short: int
    above_ma_medium: int

    price_acceleration: float
    volume_ma_ratio: float

    rsi_above_60: int
    above_bollinger_mid: int


# =====================
# FEATURE ORDER
# =====================

FEATURE_COLUMNS = [

    "price_change_1",
    "price_change_3",
    "price_change_6",

    "volume_change_1",
    "volume_change_3",
    "volume_change_6",

    "distance_ma_long",

    "volatility_short",
    "volatility_long",

    "momentum_10",

    "volume_zscore",
    "volume_spike",

    "bollinger_width",
    "bollinger_breakout",

    "rsi",

    "relative_volume",
    "breakout_strength",
    "green_candle_count",

    "volume_acceleration",
    "rsi_change",

    "above_ma_short",
    "above_ma_medium",

    "price_acceleration",
    "volume_ma_ratio",

    "rsi_above_60",
    "above_bollinger_mid"
]


# =====================
# HOME ROUTE
# =====================

@app.get("/")
def home():

    return {
        "message":
        "Crypto Pump Detector API Running"
    }


# =====================
# HEALTH CHECK
# =====================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =====================
# PREDICT ROUTE
# =====================

@app.post("/predict")
def predict(data: PumpFeatures):
    

    input_df = pd.DataFrame(
        [data.model_dump()]
    )

    input_df = input_df[
        FEATURE_COLUMNS
    ]

    probability = float(
        model.predict_proba(
            input_df
        )[0][1]
    )

    prediction = (
        probability >= 0.70
    )

    return {

        "pump_probability":
            round(
                probability * 100,
                4
            ),

        "pump_alert":
            bool(prediction)
    }