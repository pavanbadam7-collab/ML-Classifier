# Classical ML Ranking for Claims Prioritization

## 1. Objective

The primary business objective is not simply maximizing recovery value, but maximizing recovery captured within an analyst’s limited 8-hour workday.

A claim with moderate recovery potential but low handling time may be more valuable operationally than a large claim requiring several hours.

Therefore, the ranking objective optimized:

recovery efficiency = predicted recovery / handle time

This aligns the ranking system with operational productivity.

---

## 2. Data Insights

Several patterns were identified during exploratory analysis:

1. Claims with larger underpayment amounts generally had higher recovered amounts.

2. Claims with high days_since_payment showed lower recovery likelihood, suggesting timeliness is important.

3. Certain payer types and contract versions appeared to correlate with stronger recovery outcomes.

4. Analysts could only process a small subset of daily claims within the 8-hour limit, making prioritization critical.

---

## 3. Model Design

### Baseline

The baseline ranked claims using:
- largest payment variance first

This approach ignores:
- analyst handling time
- payer patterns
- claim complexity

### Pointwise Ranking Model

A RandomForestRegressor was trained to predict:
- recovered_amount

The model used engineered features including:
- underpayment amount
- claim complexity
- recovery efficiency
- authorization risk
- timeliness risk

### Time-Aware Ranking

Final ranking score:

score = predicted_recovery / handle_time_hrs

This prioritizes claims with:
- high expected recovery
- lower handling effort

---

## 4. Evaluation

Evaluation used:
- Leave-One-Worklist-Out Cross Validation

Metrics:
- NDCG@5
- NDCG@10

Results:
- Average NDCG@5 = 0.61
- Average NDCG@10 = 0.69

The model consistently outperformed naive ranking approaches by better prioritizing high-value recoverable claims near the top.

Recovery captured curves also demonstrated that the model captures a large portion of recoverable value within the first few ranked claims.

---

## 5. Limitations

The model has several limitations:

1. Relevance labels are derived from recovered amount and may not fully capture operational priorities.

2. Synthetic data may not reflect real payer behavior.

3. The current approach does not explicitly model uncertainty or denial probability.

4. Pairwise or listwise ranking approaches may further improve ranking quality.

---

## 6. Future Improvements

Given additional time and data, the following improvements would be explored:

- Pairwise learning-to-rank models
- Gradient boosting ranking objectives
- Better temporal modeling
- Analyst-specific personalization
- Explicit optimization under hourly constraints