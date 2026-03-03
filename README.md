# March Mania Risk Engine (2026)

Probabilistic forecasting pipeline for Kaggle's March Machine Learning Mania 2026 (Men + Women).

## What this repo contains
- Separate pipelines for Men's and Women's games
- Season-level feature engineering
- Tournament matchup training set creation
- Rolling season cross-validation (Brier score)
- Baseline logistic regression models
- Kaggle submission generator

## Current baseline (Rolling Season CV, last 10 seasons)
- Men Avg Brier: 0.2139
- Women Avg Brier: 0.1864

## How to run (local)
1. Put Kaggle CSVs into `data/raw/`
2. Create features:
   - `python -m src.features_team_season`
   - `python -m src.build_training`
3. Validate:
   - `python -m src.run_cv`
4. Train:
   - `python -m src.train_baseline`
5. Create submission:
   - `python -m src.make_submission`

## Roadmap
- Add NCAA seed features
- Add Elo ratings
- Add ranking features (Massey, Men)
- LightGBM + calibration + model blending
