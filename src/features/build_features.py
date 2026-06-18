import pandas as pd
import numpy as np

from src.utils.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    RSI_WINDOW,
    MA_SHORT,
    MA_MEDIUM,
    MA_LONG,
    VOLATILITY_SHORT,
    VOLATILITY_LONG,
    PUMP_LOOKAHEAD,
    DUMP_LOOKAHEAD,
    PUMP_PRICE_THRESHOLD,
    PUMP_VOLUME_THRESHOLD,
    DUMP_THRESHOLD
)


def load_data():

    df = pd.read_csv(RAW_DATA_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms"
    )

    df = df.sort_values(
        ["coin", "timestamp"]
    ).reset_index(drop=True)

    return df


def create_features(df):

    grouped = []

    for coin, coin_df in df.groupby("coin"):

        coin_df = coin_df.copy()

        # PRICE FEATURES

        coin_df["price_change_1"] = (
            coin_df["price"].pct_change() * 100
        )

        coin_df["price_change_3"] = (
            coin_df["price"].pct_change(3) * 100
        )

        coin_df["price_change_6"] = (
            coin_df["price"].pct_change(6) * 100
        )

        # VOLUME FEATURES

        coin_df["volume_change_1"] = (
            coin_df["volume"].pct_change() * 100
        )

        coin_df["volume_change_3"] = (
            coin_df["volume"].pct_change(3) * 100
        )

        coin_df["volume_change_6"] = (
            coin_df["volume"].pct_change(6) * 100
        )

        # MOVING AVERAGES

        coin_df["ma_short"] = (
            coin_df["price"]
            .rolling(MA_SHORT)
            .mean()
        )

        coin_df["ma_medium"] = (
            coin_df["price"]
            .rolling(MA_MEDIUM)
            .mean()
        )

        coin_df["ma_long"] = (
            coin_df["price"]
            .rolling(MA_LONG)
            .mean()
        )

        # BOLLINGER BANDS

        rolling_std = (
            coin_df["price"]
            .rolling(MA_LONG)
            .std()
        )

        coin_df["bollinger_upper"] = (
            coin_df["ma_long"]
            + (2 * rolling_std)
        )

        coin_df["bollinger_lower"] = (
            coin_df["ma_long"]
            - (2 * rolling_std)
        )

        coin_df["bollinger_width"] = (
            (
                coin_df["bollinger_upper"]
                - coin_df["bollinger_lower"]
            )
            / coin_df["ma_long"]
        ) * 100

        coin_df["bollinger_breakout"] = (
            coin_df["price"]
            > coin_df["bollinger_upper"]
        ).astype(int)

        # TREND DISTANCE

        coin_df["distance_ma_long"] = (
            (
                coin_df["price"]
                - coin_df["ma_long"]
            )
            / coin_df["ma_long"]
        ) * 100

        # VOLATILITY

        coin_df["volatility_short"] = (
            coin_df["price_change_1"]
            .rolling(VOLATILITY_SHORT)
            .std()
        )

        coin_df["volatility_long"] = (
            coin_df["price_change_1"]
            .rolling(VOLATILITY_LONG)
            .std()
        )

        # MOMENTUM

        coin_df["momentum_10"] = (
            coin_df["price"]
            - coin_df["price"].shift(10)
        )

        # VOLUME FEATURES

        rolling_volume_mean = (
            coin_df["volume"]
            .rolling(MA_LONG)
            .mean()
        )

        rolling_volume_std = (
            coin_df["volume"]
            .rolling(MA_LONG)
            .std()
        )

        coin_df["volume_zscore"] = (
            (
                coin_df["volume"]
                - rolling_volume_mean
            )
            / rolling_volume_std
        )

        coin_df["volume_spike"] = (
            coin_df["volume_zscore"] > 2
        ).astype(int)

        # RELATIVE VOLUME

        coin_df["relative_volume"] = (
            coin_df["volume"]
            / rolling_volume_mean
        )

        # BREAKOUT STRENGTH

        rolling_high = (
            coin_df["price"]
            .rolling(24)
            .max()
        )

        coin_df["breakout_strength"] = (
            (
                coin_df["price"]
                - rolling_high
            )
            / rolling_high
        ) * 100

        # GREEN CANDLE COUNT

        coin_df["green_candle"] = (
            coin_df["price_change_1"] > 0
        ).astype(int)

        coin_df["green_candle_count"] = (
            coin_df["green_candle"]
            .rolling(6)
            .sum()
        )

        # RSI

        delta = coin_df["price"].diff()

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
            .rolling(RSI_WINDOW)
            .mean()
        )

        avg_loss = (
            losses
            .rolling(RSI_WINDOW)
            .mean()
        )

        rs = avg_gain / avg_loss

        coin_df["rsi"] = (
            100 - (100 / (1 + rs))
        )

        # NEW FEATURES

        coin_df["volume_acceleration"] = (
            coin_df["volume_change_1"]
            - coin_df["volume_change_3"]
        )

        coin_df["rsi_change"] = (
            coin_df["rsi"]
            - coin_df["rsi"].shift(3)
        )

        coin_df["above_ma_short"] = (
            coin_df["price"]
            > coin_df["ma_short"]
        ).astype(int)

        coin_df["above_ma_medium"] = (
            coin_df["price"]
            > coin_df["ma_medium"]
        ).astype(int)

        coin_df["price_acceleration"] = (
            coin_df["price_change_1"]
            - coin_df["price_change_3"]
        )

        coin_df["rsi_above_60"] = (
            coin_df["rsi"] > 60
        ).astype(int)

        coin_df["volume_ma_ratio"] = (
            coin_df["volume"]
            /
            coin_df["volume"]
            .rolling(10)
            .mean()
        )

        coin_df["above_bollinger_mid"] = (
            coin_df["price"]
            >
            coin_df["ma_long"]
        ).astype(int)

        grouped.append(
            coin_df
        )

    return pd.concat(
        grouped,
        ignore_index=True
    )


def create_target(df):

    grouped = []

    for coin, coin_df in df.groupby("coin"):

        coin_df = coin_df.copy()

        future_price = (
            coin_df["price"]
            .shift(-PUMP_LOOKAHEAD)
        )

        future_volume = (
            coin_df["volume"]
            .shift(-PUMP_LOOKAHEAD)
        )

        dump_price = (
            coin_df["price"]
            .shift(-DUMP_LOOKAHEAD)
        )

        future_return = (
            (
                future_price
                - coin_df["price"]
            )
            / coin_df["price"]
        ) * 100

        future_volume_return = (
            (
                future_volume
                - coin_df["volume"]
            )
            / coin_df["volume"]
        ) * 100

        dump_return = (
            (
                dump_price
                - future_price
            )
            / future_price
        ) * 100

        coin_df["pump_target"] = np.where(
            (
                future_return
                > PUMP_PRICE_THRESHOLD
            )
            &
            (
                future_volume_return
                > PUMP_VOLUME_THRESHOLD
            )
            &
            (
                dump_return
                < DUMP_THRESHOLD
            ),
            1,
            0
        )

        grouped.append(
            coin_df
        )

    return pd.concat(
        grouped,
        ignore_index=True
    )


def clean_data(df):

    print("Before clean:", len(df))

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    print(
        "Rows containing NaN:",
        df.isna().any(axis=1).sum()
    )

    df = df.dropna().reset_index(drop=True)

    print("After clean:", len(df))

    return df


def save_data(df):

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print(
        f"\nProcessed data saved to:"
        f"\n{PROCESSED_DATA_PATH}"
    )


if __name__ == "__main__":

    print("Loading raw data...")

    df = load_data()

    print(
        f"Loaded {len(df)} rows"
    )

    print(
        "Creating features..."
    )

    df = create_features(df)

    print(
        "Creating targets..."
    )

    df = create_target(df)

    print(
        "Cleaning dataset..."
    )

    df = clean_data(df)

    print(
        "\nPump Target Distribution:"
    )

    print(
        df["pump_target"]
        .value_counts()
    )

    save_data(df)

    print(
        f"\nFinal Shape: {df.shape}"
    )
