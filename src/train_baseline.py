from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
import joblib

PROCESSED = Path("data/processed")

FEATURE_COLS = [
    "Diff_WinPct",
    "Diff_AvgPF",
    "Diff_AvgPA",
    "Diff_AvgScoreDiff",
    "Diff_NeutralRate",
]


def train_and_evaluate(prefix: str):
    df = pd.read_csv(PROCESSED / f"train_{prefix}.csv")

    X = df[FEATURE_COLS]
    y = df["Label"]

    # simple random split (later we improve to time-based split)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_val)[:, 1]
    brier = brier_score_loss(y_val, preds)

    print(f"{prefix.upper()} Brier Score:", round(brier, 5))

    joblib.dump(model, PROCESSED / f"model_{prefix}.pkl")


def main():
    train_and_evaluate("m")
    train_and_evaluate("w")


if __name__ == "__main__":
    main()
