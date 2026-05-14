import numpy as np


def create_features(df):
    """
    Create engineered features for ranking model.
    """

    # Underpayment amount
    df["underpayment_amount"] = (
        df["expected_payment"] - df["actual_payment"]
    )

    # Absolute payment variance
    df["abs_payment_variance"] = (
        df["payment_variance"].abs()
    )

    # Recovery efficiency
    df["expected_recovery_per_hour"] = (
        df["underpayment_amount"] /
        (df["handle_time_hrs"] + 1e-5)
    )

    # Complexity score
    df["claim_complexity"] = (
        df["num_diagnoses"] +
        df["num_procedures"]
    )

    # Timeliness risk
    df["timeliness_risk"] = np.where(
        df["days_since_payment"] > 90,
        1,
        0
    )

    # Authorization flag
    df["auth_risk"] = np.where(
        df["has_prior_auth"] == 1,
        0,
        1
    )

    return df