from pathlib import Path
import pandas as pd
from src.ingest import load_csv

PROCESSED = Path("data/processed")


FEATURE_COLS = ["WinPct", "AvgPF", "AvgPA", "AvgScoreDiff", "NeutralRate"]


def make_training(prefix: str) -> pd.DataFrame:
    """
    prefix: 'M' or 'W'
    Builds training data from NCAA tournament compact results.
    """
    tourney = load_csv(f"{prefix}NCAATourneyCompactResults.csv")
    feats = pd.read_csv(PROCESSED / f"team_season_{prefix.lower()}.csv")

    # Only keep games where we have season features for both teams
    # Winner perspective
    a = tourney[["Season", "WTeamID", "LTeamID"]].copy()
    a.rename(columns={"WTeamID": "TeamA", "LTeamID": "TeamB"}, inplace=True)
    a["Label"] = 1

    # Loser perspective (swap to create balance)
    b = tourney[["Season", "WTeamID", "LTeamID"]].copy()
    b.rename(columns={"LTeamID": "TeamA", "WTeamID": "TeamB"}, inplace=True)
    b["Label"] = 0

    pairs = pd.concat([a, b], ignore_index=True)

    # Join season features for each team
    fa = feats.rename(columns={"TeamID": "TeamA"})
    fb = feats.rename(columns={"TeamID": "TeamB"})

    df = pairs.merge(fa, on=["Season", "TeamA"], how="inner", suffixes=("", ""))
    df = df.merge(fb, on=["Season", "TeamB"], how="inner", suffixes=("_A", "_B"))

    # Build difference features
    for c in FEATURE_COLS:
        df[f"Diff_{c}"] = df[f"{c}_A"] - df[f"{c}_B"]

    keep = ["Season", "TeamA", "TeamB", "Label"] + [f"Diff_{c}" for c in FEATURE_COLS]
    return df[keep]


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    train_m = make_training("M")
    train_w = make_training("W")

    train_m.to_csv(PROCESSED / "train_m.csv", index=False)
    train_w.to_csv(PROCESSED / "train_w.csv", index=False)

    print("Saved:", PROCESSED / "train_m.csv", train_m.shape)
    print("Saved:", PROCESSED / "train_w.csv", train_w.shape)
    print(train_m.head())


if __name__ == "__main__":
    main()
