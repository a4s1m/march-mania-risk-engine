from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


@dataclass
class CVResult:
    avg_brier: float
    per_season: pd.DataFrame


def rolling_season_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "Label",
    season_col: str = "Season",
    min_train_seasons: int = 5,
    last_n_val_seasons: int = 10,
) -> CVResult:
    """
    Rolling season CV:
    - for each validation season Y, train on all seasons < Y
    - validate on season == Y
    - reports Brier score per season and average over evaluated seasons

    min_train_seasons: minimum number of distinct seasons required before first eval
    last_n_val_seasons: evaluate only the most recent N seasons (stability + speed)
    """
    df = df.copy()
    seasons = sorted(df[season_col].unique())

    # choose validation seasons (most recent N)
    val_seasons = seasons[-last_n_val_seasons:] if last_n_val_seasons else seasons

    rows = []
    for y in val_seasons:
        train = df[df[season_col] < y]
        val = df[df[season_col] == y]

        if val.empty:
            continue

        train_seasons = sorted(train[season_col].unique())
        if len(train_seasons) < min_train_seasons:
            continue

        X_train = train[feature_cols]
        y_train = train[label_col]
        X_val = val[feature_cols]
        y_val = val[label_col]

        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)

        p = model.predict_proba(X_val)[:, 1]
        brier = brier_score_loss(y_val, p)

        rows.append(
            {
                "ValSeason": int(y),
                "TrainSeasons": len(train_seasons),
                "TrainRows": int(len(train)),
                "ValRows": int(len(val)),
                "Brier": float(brier),
            }
        )

    per_season = pd.DataFrame(rows).sort_values("ValSeason").reset_index(drop=True)
    avg_brier = float(per_season["Brier"].mean()) if not per_season.empty else np.nan
    return CVResult(avg_brier=avg_brier, per_season=per_season)
