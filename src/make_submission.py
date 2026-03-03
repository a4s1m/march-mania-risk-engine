from pathlib import Path
import pandas as pd
import joblib

PROCESSED = Path("data/processed")
RAW = Path("data/raw")

FEATURE_COLS = [
    "Diff_WinPct",
    "Diff_AvgPF",
    "Diff_AvgPA",
    "Diff_AvgScoreDiff",
    "Diff_NeutralRate",
]


def load_features(prefix: str):
    return pd.read_csv(PROCESSED / f"team_season_{prefix}.csv")


def main():
    sub = pd.read_csv(RAW / "SampleSubmissionStage2.csv")

    model_m = joblib.load(PROCESSED / "model_m.pkl")
    model_w = joblib.load(PROCESSED / "model_w.pkl")

    feats_m = load_features("m")
    feats_w = load_features("w")

    preds = []

    for _, row in sub.iterrows():
        season, a, b = row["ID"].split("_")
        season = int(season)
        a = int(a)
        b = int(b)

        # determine league by team id
        if a < 2000:
            feats = feats_m
            model = model_m
        else:
            feats = feats_w
            model = model_w

        fa = feats[(feats.Season == season) & (feats.TeamID == a)]
        fb = feats[(feats.Season == season) & (feats.TeamID == b)]

        if fa.empty or fb.empty:
            preds.append(0.5)
            continue

        diff = {}
        for c in ["WinPct", "AvgPF", "AvgPA", "AvgScoreDiff", "NeutralRate"]:
            diff[f"Diff_{c}"] = fa.iloc[0][c] - fb.iloc[0][c]

        X = pd.DataFrame([diff])
        p = model.predict_proba(X)[0][1]
        preds.append(p)

    sub["Pred"] = preds
    sub.to_csv("submission.csv", index=False)

    print("Saved submission.csv")
    print(sub.head())


if __name__ == "__main__":
    main()
