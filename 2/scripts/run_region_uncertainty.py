"""Cluster-bootstrap uncertainty and a frame/coreset-matched normal-bank control."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation import operating_metrics, youden_threshold
from src.memory_bank import MemoryBank, greedy_coreset
from src.runner import score_grid
from src.stats import cluster_bootstrap_ci

N_BOOT = 10000
SEED = 20261031
MATCHED_K = 1960
MATCHED_FRAMES = 10
MATCHED_SEEDS = [11, 29, 47, 71, 97, 131, 173, 211, 257, 311]
PERCENTILE = 95.0


def load_grid(path):
    with np.load(path) as z:
        return z["grid"]


def build_bank(rows, grids, *, k, seed):
    features = np.vstack([grids[row["image"]] for row in rows])
    bank = MemoryBank()
    bank.add(greedy_coreset(features, k=k, seed=seed))
    return bank


def image_scores(rows, grids, bank, contexts, *, alpha):
    return np.array([
        np.percentile(score_grid(bank.patch_scores(grids[r["image"]]), contexts[r["image"]], alpha=alpha),
                      PERCENTILE)
        for r in rows
    ])


def sample_matched_frames(clean, stripes, *, seed):
    """Draw two 10-frame banks: clean-only versus 5 clean and 5 stripes."""
    if not clean or not stripes:
        raise ValueError("clean and stripes frame lists must be non-empty")
    rng = np.random.RandomState(seed)
    clean_only = [clean[i] for i in rng.randint(0, len(clean), size=MATCHED_FRAMES)]
    mixed_clean = [clean[i] for i in rng.randint(0, len(clean), size=MATCHED_FRAMES // 2)]
    mixed_stripes = [stripes[i] for i in rng.randint(0, len(stripes), size=MATCHED_FRAMES // 2)]
    return clean_only, mixed_clean + mixed_stripes


def region_level_metrics(labels, scores, regions, stripe_labels, *, threshold):
    """One row per unique region: score=max, label=max, any-normal-positive flags."""
    order = np.argsort(regions, kind="stable")
    regions = np.asarray(regions)[order]
    labels = np.asarray(labels)[order]
    scores = np.asarray(scores)[order]
    stripe_labels = np.asarray(stripe_labels)[order]
    boundaries = np.flatnonzero(np.r_[True, regions[1:] != regions[:-1], True])
    region_scores = []
    region_labels = []
    region_stripe = []
    region_ids = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        region_ids.append(str(regions[a]))
        region_scores.append(float(np.max(scores[a:b])))
        region_labels.append(int(np.max(labels[a:b])))
        region_stripe.append(int(np.max(stripe_labels[a:b])))
    region_scores = np.array(region_scores)
    region_labels = np.array(region_labels)
    region_stripe = np.array(region_stripe)
    predictions = region_scores >= threshold
    normal = region_labels == 0
    stripe = region_stripe == 1
    result = {
        "region_ids": region_ids,
        "auroc": float(roc_auc_score(region_labels, region_scores)),
        "faulty_region_recall": float(np.mean(predictions[region_labels == 1])),
        "normal_region_any_fpr": float(np.mean(predictions[normal])),
        "stripe_region_any_fpr": float(np.mean(predictions[stripe])) if np.any(stripe) else None,
        "n_regions": len(region_labels),
        "n_normal_regions": int(normal.sum()),
        "n_stripe_regions": int(stripe.sum()),
    }
    return result, region_ids, region_scores, region_labels, region_stripe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-only", action="store_true",
                        help="Recompute only the matched control in an existing report.")
    args = parser.parse_args()
    out = Path("outputs/region")
    report_path = out / "uncertainty_report.json"
    if args.matched_only:
        if not report_path.exists():
            raise FileNotFoundError("run the full evaluation before --matched-only")
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
    rows = list(csv.DictReader(open("data/processed_region/manifest.csv", encoding="utf-8")))
    cache = Path("outputs/l2/cache")
    grids = {r["image"]: load_grid(cache / f"{r['image']}.npz") for r in rows}
    with np.load("outputs/ablation/fft_context.npz") as z:
        contexts = {key: z[key] for key in z.files}

    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    val_labels = np.array([r["type"] == "faulty" for r in val], dtype=int)
    test_labels = np.array([r["type"] == "faulty" for r in test], dtype=int)
    test_stripe = np.array([r["type"] == "good_stripes" for r in test], dtype=int)
    test_regions = np.array([r["region"] for r in test])

    if not args.matched_only:
        # Main existing configuration: all normal state frames, effective coreset 2,000.
        normal_train = [r for r in train if r["type"] != "faulty"]
        main_bank = build_bank(normal_train, grids, k=min(2000, len(normal_train) * 196), seed=42)
        main_scores = image_scores(test, grids, main_bank, contexts, alpha=0.0)
        val_scores = image_scores(val, grids, main_bank, contexts, alpha=0.0)
        threshold = youden_threshold(val_labels, val_scores)
        main_metrics = operating_metrics(test_labels, main_scores, test_stripe, threshold=threshold)

        region_stats, region_ids, region_scores, region_labels, region_stripe = region_level_metrics(
            test_labels, main_scores, test_regions, test_stripe, threshold=threshold)

        unique_regions = np.array(region_ids)

        def metric_auroc(idx):
            return roc_auc_score(region_labels[idx], region_scores[idx])

        def metric_normal_fpr(idx):
            normal = region_labels[idx] == 0
            return float(np.mean((region_scores[idx][normal] >= threshold))) if np.any(normal) else float("nan")

        def metric_stripe_fpr(idx):
            stripe = region_stripe[idx] == 1
            normal = region_labels[idx] == 0
            sel = stripe & normal
            return float(np.mean((region_scores[idx][sel] >= threshold))) if np.any(sel) else float("nan")

        def metric_recall(idx):
            faulty = region_labels[idx] == 1
            return float(np.mean((region_scores[idx][faulty] >= threshold))) if np.any(faulty) else float("nan")

        uncertainty = {
            "unit": "unique recorded region (cluster bootstrap over region-level metrics)",
            "aggregation": "max score per region; any-positive rule for FPR/recall at region level",
            "n_bootstrap": N_BOOT,
            "seed": SEED,
            "point_metrics_state_level": main_metrics,
            "point_metrics_region_level": {k: v for k, v in region_stats.items() if k != "region_ids"},
            "auroc_ci95": cluster_bootstrap_ci(metric_auroc, unique_regions, seed=SEED, n_boot=N_BOOT),
            "normal_region_any_fpr_ci95": cluster_bootstrap_ci(metric_normal_fpr, unique_regions, seed=SEED, n_boot=N_BOOT),
            "stripe_region_any_fpr_ci95": cluster_bootstrap_ci(metric_stripe_fpr, unique_regions, seed=SEED, n_boot=N_BOOT),
            "faulty_region_recall_ci95": cluster_bootstrap_ci(metric_recall, unique_regions, seed=SEED, n_boot=N_BOOT),
        }

    clean = [r for r in train if r["type"] == "good_clean"]
    stripes = [r for r in train if r["type"] == "good_stripes"]
    matched = []
    for seed in MATCHED_SEEDS:
        clean_frames, mixed_frames = sample_matched_frames(clean, stripes, seed=seed)
        clean_bank = build_bank(clean_frames, grids, k=MATCHED_K, seed=seed)
        mixed_bank = build_bank(mixed_frames, grids, k=MATCHED_K, seed=seed)
        for label, bank in [("clean_only", clean_bank), ("clean_plus_stripes", mixed_bank)]:
            # alpha=0 makes the context choice immaterial; use the gradient map.
            local_contexts = {r["image"]: np.zeros(196, dtype=np.float32) for r in rows}
            vs = image_scores(val, grids, bank, local_contexts, alpha=0.0)
            ts = image_scores(test, grids, bank, local_contexts, alpha=0.0)
            thr = youden_threshold(val_labels, vs)
            metrics = operating_metrics(test_labels, ts, test_stripe, threshold=thr)
            metrics["auroc"] = float(roc_auc_score(test_labels, ts))
            matched.append({"seed": seed, "bank": label, "n_state_frames": MATCHED_FRAMES,
                            "effective_coreset": MATCHED_K, **metrics})

    if not args.matched_only:
        report = {"uncertainty": uncertainty}

    report["matched_control"] = {
            "design": "10 state frames and 1,960 coreset features per bank; clean-only vs 5 clean plus 5 stripes",
            "frame_sampling": "with replacement; both clean-only and mixed banks are resampled",
            "seeds": MATCHED_SEEDS,
            "runs": matched,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
