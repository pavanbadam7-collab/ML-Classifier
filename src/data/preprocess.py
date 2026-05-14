import pandas as pd


def fill_missing_values(df):
    """
    Fill missing values in dataset.
    """

    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(0)

    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        df[col] = df[col].fillna("unknown")

    return df


def encode_categorical_features(df):
    """
    One-hot encode categorical columns.
    """

    categorical_cols = [
        "payer_type",
        "visit_type",
        "contract_version"
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    return df