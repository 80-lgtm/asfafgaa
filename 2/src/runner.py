import hashlib

import numpy as np
from sklearn.metrics import roc_auc_score

from src.scoring import saga_score


def score_grid(local_scores, stripe_consistency, *, alpha):
    return saga_score(np.asarray(local_scores), np.asarray(stripe_consistency), alpha=alpha)


def select_alpha(candidates):
    if not candidates:
        raise ValueError("alpha candidates cannot be empty")
    return max(
        candidates,
        key=lambda item: (roc_auc_score(item["val_labels"], item["val_scores"]), -item["alpha"]),
    )["alpha"]


def feature_cache_key(image_name, image_bytes, *, image_size):
    digest = hashlib.sha256(image_bytes).hexdigest()
    return f"{image_name}-{image_size}-{digest}.npz"
