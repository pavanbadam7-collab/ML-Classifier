# Classical ML Ranking for Healthcare Claims Prioritization

This project solves a healthcare claims ranking problem using classical machine learning techniques.

The objective is to rank disputed insurance claims within analyst worklists so that analysts prioritize:
- high recovery value claims
- claims likely to succeed
- claims with better recovery-per-hour efficiency

The solution was developed as part of a technical assessment for Ensemble Health Partners.

---

# Problem Statement

Analysts only have 8 working hours per day and cannot process every claim in their queue.

The goal is to build a ranking system that:
1. Predicts recovery potential
2. Incorporates analyst handling time
3. Maximizes total recovery captured within limited working hours

---

# Approach

## Baseline
Naive ranking by:
- largest underpayment first

## Pointwise Ranking Model
A regression model predicts:
- expected recovered amount

Model used:
- RandomForestRegressor

## Time-Aware Ranking
Final ranking score:

score = predicted_recovery / handle_time_hrs

This prioritizes claims with:
- high expected recovery
- low analyst effort

---

# Feature Engineering

Created additional business-oriented features:

- underpayment_amount
- abs_payment_variance
- expected_recovery_per_hour
- claim_complexity
- timeliness_risk
- auth_risk

---

# Evaluation

Evaluation strategy:
- Leave-One-Worklist-Out Cross Validation

Metrics:
- NDCG@5
- NDCG@10
- Recovery Captured @K

Average Results:
- NDCG@5 = 0.61
- NDCG@10 = 0.69

---

# Project Structure

```text
src/
├── data/
├── features/
├── models/
├── evaluation/
├── inference/
└── utils/
```

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Run Training

```bash
python -m src.train
```

---

# Run Prediction Pipeline

```bash
python -m src.inference.rank_current_worklists
```

---

# Run Evaluation

```bash
python -m src.evaluate
```

---

# Outputs

Generated outputs include:
- ranked_worklists.csv
- recovery_curve.png
- cross-validation metrics

---

# Future Improvements

Potential future improvements:
- Pairwise learning-to-rank models
- XGBoost ranking objectives
- Better handling of payer-specific behavior
- Time-constrained optimization
- Calibration of recovery probabilities# ML-Classifier