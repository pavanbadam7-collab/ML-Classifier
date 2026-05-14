import numpy as np
from sklearn.metrics import ndcg_score


def calculate_ndcg(y_true, y_scores, k=5):
    """
    Compute NDCG@K.
    """

    y_true = np.asarray([y_true])
    y_scores = np.asarray([y_scores])

    return ndcg_score(
        y_true,
        y_scores,
        k=k
    )


def recovery_captured_at_k(
    recovered_amounts,
    ranked_indices,
    k
):
    """
    Fraction of total recovery captured
    within top-k ranked claims.
    """

    total_recovery = np.sum(recovered_amounts)

    top_k_indices = ranked_indices[:k]

    captured_recovery = np.sum(
        recovered_amounts[top_k_indices]
    )

    if total_recovery == 0:
        return 0

    return captured_recovery / total_recovery