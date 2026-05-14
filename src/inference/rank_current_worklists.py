import pandas as pd

from src.data.load_data import (
    load_claims_data,
    load_worklists_data,
    merge_datasets
)

from src.data.preprocess import (
    fill_missing_values,
    encode_categorical_features
)

from src.features.feature_engineering import (
    create_features
)

from src.models.pointwise_ranker import (
    train_pointwise_model,
    predict_recovery
)

from src.models.time_aware_ranker import (
    create_time_aware_score
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


def main():

    claims_df = load_claims_data(
        "data/raw/worklist_claims.csv"
    )

    worklists_df = load_worklists_data(
        "data/raw/worklists.csv"
    )

    df = merge_datasets(
        claims_df,
        worklists_df
    )

    df = fill_missing_values(df)

    df = create_features(df)

    df = encode_categorical_features(df)

    historical_df = df[
        df["is_historical"] == 1
    ].copy()

    current_df = df[
        df["is_historical"] == 0
    ].copy()

    X_train = historical_df[FEATURE_COLUMNS]

    y_train = historical_df[
        "recovered_amount"
    ]

    model = train_pointwise_model(
        X_train,
        y_train
    )

    X_current = current_df[
        FEATURE_COLUMNS
    ]

    predictions = predict_recovery(
        model,
        X_current
    )

    current_df[
        "predicted_recovery"
    ] = predictions

    current_df = create_time_aware_score(
        current_df
    )

    current_df = current_df.sort_values(
        by=[
            "worklist_id",
            "ranking_score"
        ],
        ascending=[True, False]
    )

    current_df["rank"] = current_df.groupby(
        "worklist_id"
    ).cumcount() + 1

    output_columns = [
        "worklist_id",
        "rank",
        "claim_id",
        "predicted_recovery",
        "handle_time_hrs",
        "ranking_score"
    ]

    output_df = current_df[
        output_columns
    ]

    output_df.to_csv(
        "outputs/predictions/ranked_worklists.csv",
        index=False
    )

    print("\nPredictions saved successfully.\n")

    print(output_df.head(20))


if __name__ == "__main__":
    main()