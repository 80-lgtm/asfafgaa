import numpy as np


def greedy_coreset(features: np.ndarray, *, k: int, seed: int) -> np.ndarray:
    """Greedy k-center coreset subsampling (PatchCore-style)."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must be two-dimensional")
    n = features.shape[0]
    if k <= 0 or k > n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    rng = np.random.RandomState(seed)
    first = int(rng.randint(n))
    selected = [first]
    dist = np.linalg.norm(features - features[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dist))
        selected.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(features - features[nxt], axis=1))
    return features[sorted(selected)]


class MemoryBank:
    """Small in-memory nearest-neighbor feature bank for CPU experiments."""

    def __init__(self):
        self.features = None

    def add(self, features: np.ndarray):
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        if features.shape[1] == 0:
            raise ValueError("feature dimension cannot be zero")
        self.features = features if self.features is None else np.vstack([self.features, features])

    def nearest_distance(self, feature: np.ndarray) -> float:
        if self.features is None:
            raise ValueError("memory bank is empty")
        feature = np.asarray(feature, dtype=np.float32)
        if feature.ndim != 1:
            raise ValueError("feature must be one-dimensional")
        if feature.shape[0] != self.features.shape[1]:
            raise ValueError("feature dimension does not match memory bank")
        return float(np.min(np.linalg.norm(self.features - feature[None, :], axis=1)))

    def patch_scores(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        if self.features is None:
            raise ValueError("memory bank is empty")
        distances = np.linalg.norm(self.features[None, :, :] - features[:, None, :], axis=2)
        return np.min(distances, axis=1)
