# March Madness Risk & Decision Intelligence Engine

A structured analytics system for evaluating NCAA tournament matchups using historical performance, rankings, and probabilistic modeling.

## Objectives
- Predict win probabilities for tournament matchups
- Quantify upset risk and model uncertainty
- Analyze ranking system reliability
- Provide interpretable decision-support insights

## Tech Stack
- Python
- Pandas / NumPy
- Scikit-learn / LightGBM
- Streamlit (planned)
- SQL-style relational modeling

## Project Structure
- `src/` – data pipelines and modeling scripts
- `data/` – raw data (excluded from GitHub)
- `models/` – trained models
- `reports/` – documentation and metric definitions
- `app/` – interactive dashboard

## Status
In Development

# 📈 Development Progress Log

## Phase 1 — Infrastructure & Pipeline Setup

Designed and implemented a modular ML pipeline for the March Machine Learning Mania 2026 competition.

- Structured project architecture: `src/`, `data/`, `models/`, `reports/`
- Built ingestion layer for raw Kaggle datasets
- Created processed data layer separating Men’s and Women’s pipelines
- Implemented reproducible feature engineering scripts
- Integrated Git-based version control for iterative experimentation

**Objective:** Ensure clean, leakage-free, reproducible modeling workflow.

---

## Phase 2 — Baseline Feature Engineering

Constructed season-level team performance features from regular season data:

- Win Percentage  
- Average Points For / Against  
- Average Score Differential  
- Neutral-site rate  

Converted historical NCAA tournament games into structured matchup format:

- TeamA vs TeamB representation  
- Difference-based features (TeamA − TeamB)  
- Binary outcome labeling  
- Balanced dataset generation  

This aligns directly with Kaggle’s **Brier score** probability evaluation metric.

---

## Phase 3 — Baseline Modeling & Honest Validation

**Model:**  
Logistic Regression (probability-based classifier)

**Evaluation Metric:**  
Brier Score (mean squared error of predicted probabilities)

### Initial Random Split Performance

| Dataset | Brier Score |
|----------|-------------|
| Men | 0.2007 |
| Women | 0.1888 |

### Rolling Season Cross-Validation  
(Train on seasons < Y, validate on season Y)

| Dataset | Avg Brier |
|----------|-----------|
| Men | 0.2139 |
| Women | 0.1864 |

### Key Observations

- Random splits overestimated performance (future leakage effect).
- Rolling CV revealed instability in upset-heavy seasons (2021–2022).
- Women’s model currently shows stronger year-to-year stability.
- Baseline season aggregates alone are insufficient for medal-level performance.

---

