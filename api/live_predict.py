

import requests
import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI
from pydantic import BaseModel

from pathlib import Path
app = FastAPI(
    title="Live Crypto Pump Detector"
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgboost.pkl"

model = joblib.load(MODEL_PATH)

COINS = [
    "bitcoin",
    "ethereum",
    "solana",
    "dogecoin",
    "cardano",
    "avalanche-2",
    "chainlink",
    "polygon",
    "tron",
    "litecoin",
    "shiba-inu",
    "pepe",
    "floki",
    "bonk",
    "dogwifcoin",
    "render-token",
    "bittensor",
    "fetch-ai",
    "arbitrum",
    "optimism",
    "sei-network",
    "sui",
    "aptos",
    "near",
    "kaspa",
    "injective-protocol",
    "beam-2",
    "starknet"
]

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

    # PRICE FEATURES

    df["price_change_1"] = (
        df["price"].pct_change() * 100
    )

    df["price_change_3"] = (
        df["price"].pct_change(3) * 100
    )

    df["price_change_6"] = (
        df["price"].pct_change(6) * 100
    )

    # VOLUME FEATURES

    df["volume_change_1"] = (
        df["volume"].pct_change() * 100
    )

    df["volume_change_3"] = (
        df["volume"].pct_change(3) * 100
    )

    df["volume_change_6"] = (
        df["volume"].pct_change(6) * 100
    )

    # MOVING AVERAGES

    df["ma_short"] = (
        df["price"]
        .rolling(5)
        .mean()
    )

    df["ma_medium"] = (
        df["price"]
        .rolling(10)
        .mean()
    )

    df["ma_long"] = (
        df["price"]
        .rolling(20)
        .mean()
    )

    # BOLLINGER

    rolling_std = (
        df["price"]
        .rolling(20)
        .std()
    )

    df["bollinger_upper"] = (
        df["ma_long"]
        + (2 * rolling_std)
    )

    df["bollinger_lower"] = (
        df["ma_long"]
        - (2 * rolling_std)
    )

    df["bollinger_width"] = (
        (
            df["bollinger_upper"]
            - df["bollinger_lower"]
        )
        / df["ma_long"]
    ) * 100

    df["bollinger_breakout"] = (
        df["price"]
        > df["bollinger_upper"]
    ).astype(int)

    # DISTANCE

    df["distance_ma_long"] = (
        (
            df["price"]
            - df["ma_long"]
        )
        / df["ma_long"]
    ) * 100

    # VOLATILITY

    df["volatility_short"] = (
        df["price_change_1"]
        .rolling(10)
        .std()
    )

    df["volatility_long"] = (
        df["price_change_1"]
        .rolling(20)
        .std()
    )

    # MOMENTUM

    df["momentum_10"] = (
        df["price"]
        - df["price"].shift(10)
    )

    # VOLUME Z SCORE

    rolling_volume_mean = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    rolling_volume_std = (
        df["volume"]
        .rolling(20)
        .std()
    )

    df["volume_zscore"] = (
        (
            df["volume"]
            - rolling_volume_mean
        )
        / rolling_volume_std
    )

    df["volume_spike"] = (
        df["volume_zscore"] > 2
    ).astype(int)

    # RELATIVE VOLUME

    df["relative_volume"] = (
        df["volume"]
        / rolling_volume_mean
    )

    # BREAKOUT STRENGTH

    rolling_high = (
        df["price"]
        .rolling(24)
        .max()
    )

    df["breakout_strength"] = (
        (
            df["price"]
            - rolling_high
        )
        / rolling_high
    ) * 100

    # GREEN CANDLES

    df["green_candle"] = (
        df["price_change_1"] > 0
    ).astype(int)

    df["green_candle_count"] = (
        df["green_candle"]
        .rolling(6)
        .sum()
    )

    # RSI

    delta = df["price"].diff()

    gains = delta.where(
        delta > 0,
        0
    )

    losses = -delta.where(
        delta < 0,
        0
    )

    avg_gain = (
        gains
        .rolling(14)
        .mean()
    )

    avg_loss = (
        losses
        .rolling(14)
        .mean()
    )

    rs = avg_gain / avg_loss

    df["rsi"] = (
        100 - (100 / (1 + rs))
    )

    # EXTRA FEATURES

    df["volume_acceleration"] = (
        df["volume_change_1"]
        - df["volume_change_3"]
    )

    df["rsi_change"] = (
        df["rsi"]
        - df["rsi"].shift(3)
    )

    df["above_ma_short"] = (
        df["price"]
        > df["ma_short"]
    ).astype(int)

    df["above_ma_medium"] = (
        df["price"]
        > df["ma_medium"]
    ).astype(int)

    df["price_acceleration"] = (
        df["price_change_1"]
        - df["price_change_3"]
    )

    df["rsi_above_60"] = (
        df["rsi"] > 60
    ).astype(int)

    df["volume_ma_ratio"] = (
        df["volume"]
        /
        df["volume"]
        .rolling(10)
        .mean()
    )

    df["above_bollinger_mid"] = (
        df["price"]
        > df["ma_long"]
    ).astype(int)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
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

    headers = {
        "User-Agent": "CryptoPumpDetector/1.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    print(
        f"{coin} -> Status {response.status_code}"
    )

    if response.status_code != 200:
            print("=" * 50)
            print(f"COIN: {coin}")
            print(f"STATUS: {response.status_code}")
            print("RESPONSE:")
            print(response.text)
            print("=" * 50)

            raise Exception(
            f"Status {response.status_code} for {coin}"
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

    "rsi",
    "relative_volume",
    "breakout_strength",
    "green_candle_count",
    "volume_acceleration",
    "rsi_change",
    "above_ma_short",
    "above_ma_medium",
    "price_acceleration",
    "rsi_above_60",
    "volume_ma_ratio",
    "above_bollinger_mid"
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

    "rsi",
    "relative_volume",
    "breakout_strength",
    "green_candle_count",
    "volume_acceleration",
    "rsi_change",
    "above_ma_short",
    "above_ma_medium",
    "price_acceleration",
    "rsi_above_60",
    "volume_ma_ratio",
    "above_bollinger_mid"
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
            print(

            f"ERROR: {e}"
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
