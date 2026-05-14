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

from src.evaluation.cross_validation import (
    leave_one_worklist_out_cv
)


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

    results = leave_one_worklist_out_cv(df)

    print("\nCross Validation Results:\n")

    print(results)

    print("\nAverage Metrics:\n")

    print(results.mean(numeric_only=True))


if __name__ == "__main__":
    main()