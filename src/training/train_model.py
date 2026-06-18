import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.over_sampling import SMOTE

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

        "rsi"
    ]

    X = df[feature_columns]

    y = df["pump_target"]

    return X, y, feature_columns


def train_model(X_train, y_train):

    model = RandomForestClassifier(

        n_estimators=800,

        max_depth=15,

        min_samples_split=8,

        min_samples_leaf=4,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
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

    predictions = model.predict(
        X_test
    )

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
        f"\nPrecision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1 Score: {f1:.4f}"
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
        "models/random_forest.pkl"
    )

    print(
        "\nModel Saved:"
        "\nmodels/random_forest.pkl"
    )


if __name__ == "__main__":

    print(
        "Loading processed data..."
    )

    df = load_data()

    X, y, feature_columns = (
        prepare_features(df)
    )

    print(
        f"\nDataset Shape:"
        f"\n{X.shape}"
    )

    print(
        "\nOriginal Class Distribution:"
    )

    print(
        y.value_counts()
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

    print(
        "\nBefore SMOTE:"
    )

    print(
        y_train.value_counts()
    )

    smote = SMOTE(
        sampling_strategy=0.3,
        random_state=42
    )

    X_train, y_train = (
        smote.fit_resample(
            X_train,
            y_train
        )
    )

    print(
        "\nAfter SMOTE:"
    )

    print(
        y_train.value_counts()
    )

    print(
        "\nTraining Model..."
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