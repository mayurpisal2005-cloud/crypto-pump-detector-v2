import time
import requests
import pandas as pd
import os

from src.utils.config import (
    COINS,
    RAW_DATA_PATH,
    LOOKBACK_DAYS
)


def fetch_coin_data(coin):

    try:

        url = (
            f"https://api.coingecko.com/api/v3/coins/"
            f"{coin}/market_chart"
            f"?vs_currency=usd"
            f"&days={LOOKBACK_DAYS}"
        )
        headers = {
    "User-Agent": "CryptoPumpDetector/1.0",
    "x-cg-pro-api-key": os.getenv("coingecko-api-key")
}
        print(
    "API KEY FOUND:",
    os.getenv("coingecko-api-key") is not None
)




        response = requests.get(
            url,
            headers=headers,
            timeout=20
            )
        print(f"{coin} -> Status {response.status_code}")
        if response.status_code != 200:
            print(response.text[:500])

            return pd.DataFrame()

        data = response.json()

        prices = data["prices"]
        volumes = data["total_volumes"]

        rows = []

        for i in range(len(prices)):

            rows.append(
                {
                    "coin": coin,
                    "timestamp": prices[i][0],
                    "price": prices[i][1],
                    "volume": volumes[i][1]
                }
            )

        return pd.DataFrame(rows)

    except Exception as e:

        print(
            f"Error fetching {coin}: {e}"
        )

        return pd.DataFrame()


def fetch_market_data():

    all_data = []

    total_coins = len(COINS)

    for index, coin in enumerate(
        COINS,
        start=1
    ):

        print(
            f"\n[{index}/{total_coins}] Fetching {coin}..."
        )

        df = fetch_coin_data(
            coin
        )

        if not df.empty:

            print(
                f"Collected {len(df)} rows"
            )

            all_data.append(df)

        else:

            print(
                f"Skipped {coin}"
            )

        # Avoid CoinGecko rate limits
        time.sleep(3)

    if len(all_data) == 0:

        raise Exception(
            "No data collected from any coin."
        )

    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    return final_df


def save_data(df):

    df.to_csv(
        RAW_DATA_PATH,
        index=False
    )

    print("\n======================")
    print(f"Saved {len(df)} rows")
    print(f"Path: {RAW_DATA_PATH}")
    print("======================")


if __name__ == "__main__":

    print("\n======================")
    print("STARTING DATA COLLECTION")
    print("======================")

    print(
        f"\nCoins configured: {len(COINS)}"
    )

    print(
        f"Lookback days: {LOOKBACK_DAYS}"
    )

    df = fetch_market_data()

    print("\n======================")
    print("DATASET PREVIEW")
    print("======================")

    print(df.head())

    print("\n======================")
    print("ROWS PER COIN")
    print("======================")

    print(
        df["coin"].value_counts()
    )

    save_data(df)