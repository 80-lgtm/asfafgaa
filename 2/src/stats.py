import numpy as np


def paired_bootstrap_ci(a, b, *, seed, n_boot=10000, alpha=0.05):
    """Percentile bootstrap CI for mean(a - b) over paired samples."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("inputs must be equal-length one-dimensional arrays")
    if len(a) == 0:
        raise ValueError("inputs cannot be empty")
    rng = np.random.RandomState(seed)
    diffs = a - b
    means = np.empty(n_boot)
    n = len(diffs)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        means[i] = diffs[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def cluster_bootstrap_indices(clusters, *, seed, n_boot):
    """Yield row indices for each bootstrap replicate, keeping a cluster intact."""
    clusters = np.asarray(clusters)
    if clusters.ndim != 1 or len(clusters) == 0:
        raise ValueError("clusters must be a non-empty one-dimensional array")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    unique = np.unique(clusters)
    members = {value: np.flatnonzero(clusters == value) for value in unique}
    rng = np.random.RandomState(seed)
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        yield np.concatenate([members[value] for value in sampled])


def cluster_bootstrap_ci(metric, clusters, *, seed, n_boot, alpha=0.05):
    """Percentile CI for a metric, reporting undefined replicates separately."""
    values = []
    skipped = 0
    for idx in cluster_bootstrap_indices(clusters, seed=seed, n_boot=n_boot):
        try:
            value = float(metric(idx))
        except (ValueError, ZeroDivisionError):
            skipped += 1
            continue
        if np.isfinite(value):
            values.append(value)
        else:
            skipped += 1
    if not values:
        raise ValueError("all bootstrap replicates were undefined")
    values = np.asarray(values)
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ci95": [float(lo), float(hi)], "n_defined": len(values), "n_skipped": skipped}


def region_any_fpr(labels, scores, clusters, *, threshold, group=None):
    """Region-level normal FPR; a normal region is positive if any selected row is."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    clusters = np.asarray(clusters)
    if not (labels.shape == scores.shape == clusters.shape) or labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels, scores, and clusters must be matching one-dimensional arrays")
    selected = scores >= threshold
    if group is not None:
        group = np.asarray(group, dtype=bool)
        if group.shape != labels.shape:
            raise ValueError("group must have the same shape as labels")
        if np.any(group & (labels != 0)):
            raise ValueError("group can only select normal rows")
        selected &= group
    hits = 0
    total = 0
    for region in np.unique(clusters[labels == 0]):
        in_region = clusters == region
        total += 1
        if np.any(selected & in_region & (labels == 0)):
            hits += 1
    return float(hits / total) if total else float("nan")
