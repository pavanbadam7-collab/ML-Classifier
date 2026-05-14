def create_time_aware_score(df):
    """
    Ranking score:
    predicted recovery per hour.
    """

    df["ranking_score"] = (
        df["predicted_recovery"] /
        (df["handle_time_hrs"] + 1e-5)
    )

    return df