import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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


def recovery_curve(df):

    recovery_rates = []

    total_recovery = df[
        "recovered_amount"
    ].sum()

    for k in range(1, 11):

        top_k = df.head(k)

        captured = top_k[
            "recovered_amount"
        ].sum()

        recovery_rates.append(
            captured / total_recovery
        )

    return recovery_rates


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

    X_train = historical_df[
        FEATURE_COLUMNS
    ]

    y_train = historical_df[
        "recovered_amount"
    ]

    model = train_pointwise_model(
        X_train,
        y_train
    )

    predictions = predict_recovery(
        model,
        X_train
    )

    historical_df[
        "predicted_recovery"
    ] = predictions

    historical_df = create_time_aware_score(
        historical_df
    )

    ranked_df = historical_df.sort_values(
        by="ranking_score",
        ascending=False
    )

    curve = recovery_curve(
        ranked_df
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, 11),
        curve,
        marker="o"
    )

    plt.xlabel("Top-K Claims Worked")

    plt.ylabel(
        "Fraction of Recovery Captured"
    )

    plt.title(
        "Recovery Captured @K"
    )

    plt.grid(True)

    plt.savefig(
        "outputs/figures/recovery_curve.png"
    )

    plt.close()

    print(
        "\nRecovery curve saved.\n"
    )


if __name__ == "__main__":
    main()