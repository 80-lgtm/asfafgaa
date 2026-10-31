import numpy as np


def saga_score(local_scores: np.ndarray, stripe_consistency: np.ndarray, *, alpha: float) -> np.ndarray:
    """Suppress stripe-like local anomalies; preserve zero scores exactly."""
    local_scores = np.asarray(local_scores, dtype=np.float32)
    stripe_consistency = np.asarray(stripe_consistency, dtype=np.float32)
    if local_scores.shape != stripe_consistency.shape:
        raise ValueError("local scores and stripe consistency must have the same shape")
    if local_scores.ndim != 1:
        raise ValueError("scores must be one-dimensional patch score arrays")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    denominator = 1.0 + alpha * stripe_consistency
    return local_scores / denominator
