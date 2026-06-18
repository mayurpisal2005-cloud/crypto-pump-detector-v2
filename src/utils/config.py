# ==========================
# ASSET SETTINGS
# ==========================

COINS = [

    # Large Caps
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

    # Meme Coins
    "shiba-inu",
    "pepe",
    "floki",
    "bonk",
    "dogwifcoin",

    # AI Coins
    "render-token",
    "bittensor",
    "fetch-ai",

    # Volatile Mid Caps
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
# DATA SETTINGS
# ==========================

RAW_DATA_PATH = "data/raw/btc_hourly.csv"

PROCESSED_DATA_PATH = (
    "data/processed/features.csv"
)

LOOKBACK_DAYS = 365


# ==========================
# FEATURE ENGINEERING
# ==========================

RSI_WINDOW = 14

MA_SHORT = 5
MA_MEDIUM = 10
MA_LONG = 20

VOLATILITY_SHORT = 10
VOLATILITY_LONG = 20


# ==========================
# PUMP LABEL RULES
# ==========================

PUMP_LOOKAHEAD = 6

DUMP_LOOKAHEAD = 12

PUMP_PRICE_THRESHOLD = 1.5

PUMP_VOLUME_THRESHOLD = 20.0

DUMP_THRESHOLD = -0.5


# ==========================
# TRAINING
# ==========================

TEST_SIZE = 0.2

RANDOM_STATE = 42

SMOTE_RATIO = 0.3

