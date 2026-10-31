import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def binary_metrics(labels, scores, stripe_labels=None) -> dict[str, float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    unique = set(np.unique(labels).tolist())
    if unique != {0, 1}:
        raise ValueError("binary metrics require exactly two classes")
    result = {"auroc": float(roc_auc_score(labels, scores))}
    if stripe_labels is not None:
        stripe = np.asarray(stripe_labels)
        if stripe.shape != labels.shape:
            raise ValueError("stripe labels must have the same shape as labels")
        stripe_normals = stripe == 1
        if np.any(stripe_normals & (labels != 0)):
            raise ValueError("stripe labels must indicate normal samples only")
        if np.any(stripe_normals):
            threshold = float(np.min(scores[labels == 1]))
            result["stripe_fpr"] = float(np.mean(scores[stripe_normals] >= threshold))
    return result


def youden_threshold(labels, scores) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be equal-length one-dimensional arrays")
    unique_scores = np.unique(scores)
    candidates = (unique_scores[:-1] + unique_scores[1:]) / 2
    candidates = np.concatenate(([unique_scores[0] - 1e-6], candidates, [unique_scores[-1] + 1e-6]))
    positives = labels == 1
    negatives = labels == 0
    best_j = -np.inf
    best_thresholds = []
    for threshold in candidates:
        predicted = scores >= threshold
        tpr = np.mean(predicted[positives])
        fpr = np.mean(predicted[negatives])
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_thresholds = [threshold]
        elif j == best_j:
            best_thresholds.append(threshold)
    reference = np.median(scores[positives])
    return float(min(best_thresholds, key=lambda value: (abs(value - reference), value)))


def operating_metrics(labels, scores, stripe_labels, *, threshold) -> dict[str, float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    stripe_labels = np.asarray(stripe_labels)
    if labels.shape != scores.shape or labels.shape != stripe_labels.shape:
        raise ValueError("labels, scores, and stripe labels must have matching shapes")
    predictions = scores >= threshold
    positives = labels == 1
    normals = labels == 0
    stripes = stripe_labels == 1
    return {
        "threshold": float(threshold),
        "faulty_recall": float(np.mean(predictions[positives])),
        "stripe_fpr": float(np.mean(predictions[stripes])),
        "normal_fpr": float(np.mean(predictions[normals])),
    }
