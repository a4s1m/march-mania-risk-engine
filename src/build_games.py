from pathlib import Path
import pandas as pd
from src.ingest import load_csv

OUT_DIR = Path("data/processed")

def build_games_compact(prefix: str) -> pd.DataFrame:
    # prefix: "M" or "W"
    df = load_csv(f"{prefix}RegularSeasonCompactResults.csv")
    df["ScoreDiff"] = df["WScore"] - df["LScore"]
    df["IsNeutral"] = (df["WLoc"] == "N").astype(int)
    df["Prefix"] = prefix
    return df

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    m = build_games_compact("M")
    w = build_games_compact("W")

    m.to_csv(OUT_DIR / "games_m_compact.csv", index=False)
    w.to_csv(OUT_DIR / "games_w_compact.csv", index=False)

    print("Saved men:", (OUT_DIR / "games_m_compact.csv"), m.shape)
    print("Saved women:", (OUT_DIR / "games_w_compact.csv"), w.shape)
    print(m.head())
