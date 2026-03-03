from pathlib import Path
import pandas as pd

PROCESSED = Path("data/processed")


def team_season_features_from_compact(games: pd.DataFrame) -> pd.DataFrame:
    # Convert each game to two team-perspective rows (long format)
    w = games[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc", "NumOT", "IsNeutral"]].copy()
    l = games[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc", "NumOT", "IsNeutral"]].copy()

    w.rename(columns={"WTeamID": "TeamID", "LTeamID": "OppTeamID"}, inplace=True)
    w["PointsFor"] = w["WScore"]
    w["PointsAgainst"] = w["LScore"]
    w["IsWin"] = 1

    l.rename(columns={"LTeamID": "TeamID", "WTeamID": "OppTeamID"}, inplace=True)
    l["PointsFor"] = l["LScore"]
    l["PointsAgainst"] = l["WScore"]
    l["IsWin"] = 0

    long_df = pd.concat([w, l], ignore_index=True)
    long_df["ScoreDiff"] = long_df["PointsFor"] - long_df["PointsAgainst"]

    agg = (
        long_df.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            Games=("IsWin", "count"),
            Wins=("IsWin", "sum"),
            WinPct=("IsWin", "mean"),
            AvgPF=("PointsFor", "mean"),
            AvgPA=("PointsAgainst", "mean"),
            AvgScoreDiff=("ScoreDiff", "mean"),
            NeutralRate=("IsNeutral", "mean"),
        )
    )
    return agg


def build_and_save(prefix: str):
    in_path = PROCESSED / f"games_{prefix.lower()}_compact.csv"
    out_path = PROCESSED / f"team_season_{prefix.lower()}.csv"

    games = pd.read_csv(in_path)
    feats = team_season_features_from_compact(games)
    feats.to_csv(out_path, index=False)

    print("Saved:", out_path, feats.shape)
    print(feats.head())


if __name__ == "__main__":
    build_and_save("m")
    build_and_save("w")
