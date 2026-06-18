from src.utils.config import COINS
import requests
import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI
from pydantic import BaseModel


# ==========================
# LOAD MODEL
# ==========================

model = joblib.load(
    "models/xgboost.pkl"
)

app = FastAPI(
    title="Live Crypto Pump Detector"
)


# ==========================
# INPUT SCHEMA
# ==========================

class CoinRequest(BaseModel):

    coin: str


# ==========================
# RSI
# ==========================

def calculate_rsi(
    series,
    period=14
):

    delta = series.diff()

    gain = (
        delta.where(
            delta > 0,
            0
        )
    ).rolling(period).mean()

    loss = (
        -delta.where(
            delta < 0,
            0
        )
    ).rolling(period).mean()

    rs = gain / loss

    return (
        100 -
        (
            100 /
            (1 + rs)
        )
    )


# ==========================
# FEATURE ENGINEERING
# ==========================

def build_features(df):

    df["price_change_1"] = (
        df["price"]
        .pct_change(1)
        * 100
    )

    df["price_change_3"] = (
        df["price"]
        .pct_change(3)
        * 100
    )

    df["price_change_6"] = (
        df["price"]
        .pct_change(6)
        * 100
    )

    df["volume_change_1"] = (
        df["volume"]
        .pct_change(1)
        * 100
    )

    df["volume_change_3"] = (
        df["volume"]
        .pct_change(3)
        * 100
    )

    df["volume_change_6"] = (
        df["volume"]
        .pct_change(6)
        * 100
    )

    ma_long = (
        df["price"]
        .rolling(20)
        .mean()
    )

    df["distance_ma_long"] = (
        (
            df["price"]
            - ma_long
        )
        / ma_long
    ) * 100

    df["volatility_short"] = (
        df["price"]
        .pct_change()
        .rolling(10)
        .std()
        * 100
    )

    df["volatility_long"] = (
        df["price"]
        .pct_change()
        .rolling(20)
        .std()
        * 100
    )

    df["momentum_10"] = (
        df["price"]
        - df["price"].shift(10)
    )

    volume_mean = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    volume_std = (
        df["volume"]
        .rolling(20)
        .std()
    )

    df["volume_zscore"] = (
        (
            df["volume"]
            - volume_mean
        )
        / volume_std
    )

    df["volume_spike"] = np.where(
        df["volume_zscore"] > 2,
        1,
        0
    )

    rolling_mean = (
        df["price"]
        .rolling(20)
        .mean()
    )

    rolling_std = (
        df["price"]
        .rolling(20)
        .std()
    )

    upper_band = (
        rolling_mean
        + 2 * rolling_std
    )

    lower_band = (
        rolling_mean
        - 2 * rolling_std
    )

    df["bollinger_width"] = (
        (
            upper_band
            - lower_band
        )
        / rolling_mean
    ) * 100

    df["bollinger_breakout"] = np.where(
        df["price"] > upper_band,
        1,
        0
    )

    df["rsi"] = calculate_rsi(
        df["price"]
    )

    df = df.dropna()

    return df


# ==========================
# FETCH LIVE DATA
# ==========================

def fetch_coin_data(coin):

    url = (
        f"https://api.coingecko.com/api/v3/"
        f"coins/{coin}/market_chart"
        f"?vs_currency=usd&days=30"
    )

    response = requests.get(url)

    if response.status_code != 200:

        raise Exception(
            f"Coin not found: {coin}"
        )

    data = response.json()

    prices = data["prices"]
    volumes = data["total_volumes"]

    rows = []

    for i in range(len(prices)):

        rows.append(
            {
                "price":
                    prices[i][1],

                "volume":
                    volumes[i][1]
            }
        )

    return pd.DataFrame(rows)


# ==========================
# ROUTES
# ==========================

@app.get("/")
def home():

    return {
        "status": "running"
    }


@app.post("/predict-coin")
def predict_coin(
    request: CoinRequest
):

    feature_columns = [

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

        "rsi"
    ]

    df = fetch_coin_data(
        request.coin
    )

    df = build_features(
        df
    )

    latest = df.iloc[-1]

    print("\n")
    print("=" * 50)
    print(f"COIN: {request.coin.upper()}")
    print("=" * 50)

    print(
        latest[
            feature_columns
        ]
    )

    X = pd.DataFrame(
        [latest[feature_columns]]
    )

    probability = float(
        model.predict_proba(X)[0][1]
    )

    probability_percent = (
        probability * 100
    )

    ALERT_THRESHOLD = 0.10

    pump_alert = (
        probability >= ALERT_THRESHOLD
    )

    if probability_percent >= 10:

        confidence = "HIGH"

    elif probability_percent >= 5:

        confidence = "MEDIUM"

    elif probability_percent >= 2:

        confidence = "LOW"

    else:

        confidence = "NONE"

    print("\nProbability:", probability)
    print(
        "Probability (%):",
        probability_percent
    )
    print(
        "Pump Alert:",
        pump_alert
    )
    print(
        "Confidence:",
        confidence
    )
    print("=" * 50)

    return {

        "coin":
            request.coin,

        "pump_probability":
            round(
                probability_percent,
                4
            ),

        "confidence":
            confidence,

        "pump_alert":
            pump_alert
    }
@app.get("/scan-market")
def scan_market():

    results = []
    failed_coins = []

    feature_columns = [

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

        "rsi"
    ]

    print("\n")
    print("=" * 60)
    print("SCANNING MARKET")
    print("=" * 60)

    for coin in COINS:

        try:

            df = fetch_coin_data(
                coin
            )

            df = build_features(
                df
            )

            latest = df.iloc[-1]

            X = pd.DataFrame(
                [latest[feature_columns]]
            )

            probability = float(
                model.predict_proba(X)[0][1]
            ) * 100

            print(
                f"{coin}: "
                f"{round(probability, 4)}%"
            )

            if probability >= 10:

                confidence = "HIGH"

            elif probability >= 5:

                confidence = "MEDIUM"

            elif probability >= 2:

                confidence = "LOW"

            else:

                confidence = "NONE"

            results.append(
                {
                    "coin": coin,
                    "pump_probability": round(
                        float(probability),
                        4
                    ),
                    "confidence": confidence
                }
            )

        except Exception as e:

            print(
                f"Failed: {coin}"
            )

            failed_coins.append(
                coin
            )

            continue

    results = sorted(
        results,
        key=lambda x:
        x["pump_probability"],
        reverse=True
    )

    return {

        "coins_scanned":
            len(results),

        "coins_failed":
            len(failed_coins),

        "failed_coins":
            failed_coins,

        "top_pump_candidates":
            results[:10]
    }