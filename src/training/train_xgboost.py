import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.utils.config import (
    PROCESSED_DATA_PATH
)


def load_data():

    return pd.read_csv(
        PROCESSED_DATA_PATH
    )


def prepare_features(df):

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
        "above_bollinger_mid",
    ]

    X = df[feature_columns]

    y = df["pump_target"]

    return X, y, feature_columns


def train_model(X_train, y_train):

    model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,

    scale_pos_weight=82,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42
)   
    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(
    model,
    X_test,
    y_test
):

    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80
    ]

    print("\nTHRESHOLD ANALYSIS\n")

    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        print(
            f"Threshold={threshold:.2f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f}"
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    print("\n")
    print("=" * 60)
    print(f"BEST THRESHOLD: {best_threshold}")
    print(f"BEST F1: {best_f1:.4f}")
    print("=" * 60)

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    print("\nCONFUSION MATRIX\n")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print(
        "\nCLASSIFICATION REPORT\n"
    )

    print(
        classification_report(
            y_test,
            predictions
        )
    )

def show_feature_importance(
    model,
    feature_columns
):

    importance_df = pd.DataFrame(
        {
            "feature":
                feature_columns,

            "importance":
                model.feature_importances_
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        "\nFEATURE IMPORTANCE\n"
    )

    print(
        importance_df
    )


def save_model(model):

    joblib.dump(
        model,
        "models/xgboost.pkl"
    )

    print(
        "\nModel Saved:"
        "\nmodels/xgboost.pkl"
    )


if __name__ == "__main__":

    print(
        "Loading processed data..."
    )

    df = load_data()

    X, y, feature_columns = (
        prepare_features(df)
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
    )

    smote = SMOTE(
        sampling_strategy=0.15,
        random_state=42
    )

    X_train, y_train = (
        smote.fit_resample(
            X_train,
            y_train
        )
    )

    print(
        "\nTraining XGBoost..."
    )

    model = train_model(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    show_feature_importance(
        model,
        feature_columns
    )

    save_model(
        model
    )