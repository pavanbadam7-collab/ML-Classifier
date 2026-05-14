def baseline_rank(df):
    """
    Naive baseline:
    Rank by largest underpayment first.
    """

    ranked_df = df.sort_values(
        by="underpayment_amount",
        ascending=False
    ).copy()

    ranked_df["baseline_rank"] = range(
        1,
        len(ranked_df) + 1
    )

    return ranked_df