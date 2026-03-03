from pathlib import Path
import pandas as pd
from src.validate_cv import rolling_season_cv

PROCESSED = Path("data/processed")

FEATURE_COLS = [
    "Diff_WinPct",
    "Diff_AvgPF",
    "Diff_AvgPA",
    "Diff_AvgScoreDiff",
    "Diff_NeutralRate",
]

def main():
    for prefix in ["m", "w"]:
        df = pd.read_csv(PROCESSED / f"train_{prefix}.csv")
        res = rolling_season_cv(
            df,
            feature_cols=FEATURE_COLS,
            min_train_seasons=5,
            last_n_val_seasons=10,
        )
        print(f"\n{prefix.upper()} Rolling CV Avg Brier: {res.avg_brier:.5f}")
        print(res.per_season.tail(10).to_string(index=False))

if __name__ == "__main__":
    main()
