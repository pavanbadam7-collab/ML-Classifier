import pandas as pd


def load_claims_data(claims_path):
    """
    Load claims dataset.
    """
    return pd.read_csv(claims_path)


def load_worklists_data(worklists_path):
    """
    Load worklists dataset.
    """
    return pd.read_csv(worklists_path)


def merge_datasets(claims_df, worklists_df):
    """
    Merge claims and worklists data.
    """
    merged_df = claims_df.merge(
        worklists_df,
        on="worklist_id",
        how="left"
    )

    return merged_df