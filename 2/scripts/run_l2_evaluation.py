"""L2 full-region evaluation: features, coreset scoring, alpha ablation,
paired bootstrap, bbox localization, and result figures from real outputs."""
import csv
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

from src.evaluation import operating_metrics, youden_threshold
from src.features import ResNet18Backend
from src.localization import bbox_hit, parse_labels_csv
from src.memory_bank import MemoryBank, greedy_coreset
from src.runner import feature_cache_key, score_grid
from src.sadd import archive_image_path
from src.stats import paired_bootstrap_ci
from src.stripe_context import stripe_consistency_map

ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
CORESET_K = 2000
PERCENTILE = 95.0


def main():
    out = Path("outputs/l2")
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open("data/processed_full/manifest.csv", encoding="utf-8")))
    backend = ResNet18Backend(device="cpu").init_backend()
    features = {}
    with zipfile.ZipFile("data/raw/SADD_v1.1.zip") as archive:
        for i, row in enumerate(rows, 1):
            key = feature_cache_key(row["image"], b"", image_size=224)
            path = cache / f"{row['image']}.npz"
            if path.exists():
                with np.load(path) as z:
                    features[row["image"]] = {"grid": z["grid"], "context": z["context"]}
                continue
            raw = archive.read(archive_image_path(row))
            image = np.asarray(Image.open(io.BytesIO(raw)).convert("L"))
            grid, shape = backend.extract_feature_grid(image, patch_size=224)
            ctx = stripe_consistency_map(image, grid_shape=shape).reshape(-1)
            features[row["image"]] = {"grid": grid, "context": ctx}
            np.savez_compressed(path, grid=grid, context=ctx)
            if i % 100 == 0:
                print(f"features {i}/{len(rows)}", flush=True)
    train_normal = [r for r in rows if r["split"] == "train" and r["type"] != "faulty"]
    all_features = np.vstack([features[r["image"]]["grid"] for r in train_normal])
    coreset = greedy_coreset(all_features, k=min(CORESET_K, len(all_features)), seed=42)
    bank = MemoryBank()
    bank.add(coreset)
    print(f"memory bank: {bank.features.shape} (coreset of {len(all_features)})", flush=True)

    # Local scores computed once; alpha only rescales.
    local = {r["image"]: bank.patch_scores(features[r["image"]]["grid"]) for r in rows}
    ctx = {r["image"]: features[r["image"]]["context"] for r in rows}

    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    val_labels = np.array([int(r["type"] == "faulty") for r in val])
    test_labels = np.array([int(r["type"] == "faulty") for r in test])
    test_stripe = np.array([int(r["type"] == "good_stripes") for r in test])

    ablation = []
    best = None
    for alpha in ALPHAS:
        val_scores = np.array([
            np.percentile(score_grid(local[r["image"]], ctx[r["image"]], alpha=alpha), PERCENTILE)
            for r in val
        ])
        val_auroc = float(roc_auc_score(val_labels, val_scores))
        ablation.append({"alpha": alpha, "val_auroc": val_auroc})
        if best is None or val_auroc > best["val_auroc"]:
            best = {"alpha": alpha, "val_auroc": val_auroc}
        print(f"alpha={alpha} val_auroc={val_auroc:.4f}", flush=True)
    alpha_star = best["alpha"]

    def image_scores(subset, alpha):
        return np.array([
            np.percentile(score_grid(local[r["image"]], ctx[r["image"]], alpha=alpha), PERCENTILE)
            for r in subset
        ])

    results = {}
    for name, alpha in [("baseline_alpha0", 0.0), ("saga_alpha_star", alpha_star)]:
        thr = youden_threshold(val_labels, image_scores(val, alpha))
        ts = image_scores(test, alpha)
        m = operating_metrics(test_labels, ts, test_stripe, threshold=thr)
        m["auroc"] = float(roc_auc_score(test_labels, ts))
        m["alpha"] = alpha
        results[name] = {"metrics": m, "scores": ts.tolist(), "threshold": thr}
        print(f"{name}: {m}", flush=True)

    # Paired bootstrap on shared test regions: stripe FPR difference and AUROC difference.
    # FPR difference via per-region binary predictions at each method's own threshold.
    base_pred = (image_scores(test, 0.0) >= results["baseline_alpha0"]["threshold"]).astype(float)
    saga_pred = (image_scores(test, alpha_star) >= results["saga_alpha_star"]["threshold"]).astype(float)
    stripe_mask = test_stripe == 1
    fpr_lo, fpr_hi = paired_bootstrap_ci(base_pred[stripe_mask], saga_pred[stripe_mask], seed=42, n_boot=10000) \
        if stripe_mask.sum() >= 2 else (float("nan"), float("nan"))
    recall_base = base_pred[test_labels == 1]
    recall_saga = saga_pred[test_labels == 1]
    rec_lo, rec_hi = paired_bootstrap_ci(recall_saga, recall_base, seed=42, n_boot=10000)

    # bbox localization on faulty test images at alpha_star.
    with zipfile.ZipFile("data/raw/SADD_v1.1.zip") as archive:
        boxes = parse_labels_csv(io.StringIO(
            archive.read("SADD/labels/faulty/expert_annotations/labels.csv").decode("utf-8-sig")))
    hits, total = 0, 0
    for r in test:
        if r["type"] != "faulty":
            continue
        grid_scores = score_grid(local[r["image"]], ctx[r["image"]], alpha=alpha_star)
        thr_map = np.percentile(grid_scores, PERCENTILE)
        for box in boxes.get(r["image"], []):
            total += 1
            # Map full-image bbox to 14x14 grid coordinates (image is 1920x1050).
            gx0, gy0 = box[0] / 1920 * 14, box[1] / 1050 * 14
            gx1, gy1 = box[2] / 1920 * 14, box[3] / 1050 * 14
            if bbox_hit(grid_scores.reshape(14, 14), (gx0, gy0, gx1, gy1), threshold=thr):
                hits += 1
    localization = {"bbox_hit_rate": hits / total if total else None, "hits": hits, "total_boxes": total}
    print(f"localization: {localization}", flush=True)

    report = {
        "status": "completed",
        "alpha_ablation": ablation,
        "selected_alpha": alpha_star,
        "results": {k: v["metrics"] for k, v in results.items()},
        "bootstrap": {
            "stripe_fpr_diff_ci95": [fpr_lo, fpr_hi],
            "recall_diff_ci95": [rec_lo, rec_hi],
            "n_test_regions": len(test),
            "n_stripe_regions": int(stripe_mask.sum()),
        },
        "localization": localization,
        "coreset": {"k": int(len(coreset)), "from": int(len(all_features))},
    }
    with open(out / "l2_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(out / "scores.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "type", "split", "region", "score_alpha0", "score_alpha_star"])
        s0, s1 = image_scores(test, 0.0), image_scores(test, alpha_star)
        for r, a, b in zip(test, s0, s1):
            w.writerow([r["image"], r["type"], r["split"], r["region"], f"{a:.6f}", f"{b:.6f}"])

    # Figures from real outputs only.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, alpha, label in [("baseline_alpha0", 0.0, "PatchCore-like (a=0)"),
                               ("saga_alpha_star", alpha_star, f"SAGA-AD (a={alpha_star})")]:
        ts = image_scores(test, alpha)
        fpr, tpr, _ = roc_curve(test_labels, ts)
        axes[0].plot(fpr, tpr, label=f"{label} AUROC={roc_auc_score(test_labels, ts):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].legend(fontsize=8); axes[0].set_title("ROC (test, region-level)")
    axes[1].plot([a["alpha"] for a in ablation], [a["val_auroc"] for a in ablation], "o-")
    axes[1].set_xlabel("alpha"); axes[1].set_ylabel("Validation AUROC"); axes[1].set_ylim(0.9, 1.01)
    axes[1].set_title("Alpha ablation (val)")
    fig.tight_layout()
    fig.savefig(out / "figure_roc_ablation.png", dpi=200)
    fig.savefig(out / "figure_roc_ablation.pdf")
    print("figure saved", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
