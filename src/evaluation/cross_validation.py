import pandas as pd

from src.models.pointwise_ranker import (
    train_pointwise_model,
    predict_recovery
)

from src.models.time_aware_ranker import (
    create_time_aware_score
)

from src.evaluation.metrics import (
    calculate_ndcg
)


FEATURE_COLUMNS = [
    "total_billed",
    "expected_payment",
    "actual_payment",
    "payment_variance",
    "num_diagnoses",
    "num_procedures",
    "has_prior_auth",
    "days_since_payment",
    "handle_time_hrs",
    "underpayment_amount",
    "abs_payment_variance",
    "expected_recovery_per_hour",
    "claim_complexity",
    "timeliness_risk",
    "auth_risk"
]


def leave_one_worklist_out_cv(df):
    """
    Leave-one-worklist-out validation.
    """

    historical_df = df[
        df["is_historical"] == 1
    ].copy()

    worklists = historical_df[
        "worklist_id"
    ].unique()

    results = []

    for test_worklist in worklists:

        train_df = historical_df[
            historical_df["worklist_id"] != test_worklist
        ]

        test_df = historical_df[
            historical_df["worklist_id"] == test_worklist
        ]

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["recovered_amount"]

        X_test = test_df[FEATURE_COLUMNS]

        model = train_pointwise_model(
            X_train,
            y_train
        )

        predictions = predict_recovery(
            model,
            X_test
        )

        test_df["predicted_recovery"] = predictions

        test_df = create_time_aware_score(
            test_df
        )

        ndcg_5 = calculate_ndcg(
            test_df["relevance_label"],
            test_df["ranking_score"],
            k=5
        )

        ndcg_10 = calculate_ndcg(
            test_df["relevance_label"],
            test_df["ranking_score"],
            k=10
        )

        results.append({
            "worklist_id": test_worklist,
            "ndcg_5": ndcg_5,
            "ndcg_10": ndcg_10
        })

    return pd.DataFrame(results)