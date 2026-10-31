"""Leakage-free region-level evaluation on the corrected split."""
import csv
import io
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

from src.evaluation import operating_metrics, youden_threshold
from src.localization import bbox_hit, parse_labels_csv
from src.memory_bank import MemoryBank, greedy_coreset
from src.runner import score_grid
from src.sadd import archive_image_path
from src.stripe_context import frequency_consistency_map

CORESET_K = 2000
PERCENTILE = 95.0
GRAD_ALPHAS = [0.0, 1.0, 2.0, 4.0, 8.0]
FFT_ALPHAS = [0.0, 1.0, 2.0, 4.0]


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main():
    out = Path("outputs/region")
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open("data/processed_region/manifest.csv", encoding="utf-8")))
    feat_cache = Path("outputs/l2/cache")
    grids = {r["image"]: load_npz(feat_cache / f"{r['image']}.npz")["grid"] for r in rows}
    grad_ctx = {r["image"]: load_npz(feat_cache / f"{r['image']}.npz")["context"] for r in rows}
    with np.load("outputs/ablation/fft_context.npz") as z:
        fft_ctx = {k: z[k] for k in z.files}

    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    val_labels = np.array([int(r["type"] == "faulty") for r in val])
    test_labels = np.array([int(r["type"] == "faulty") for r in test])
    test_stripe = np.array([int(r["type"] == "good_stripes") for r in test])

    def build_bank(include_stripes):
        members = [r for r in train if r["type"] == "good_clean"]
        if include_stripes:
            members += [r for r in train if r["type"] == "good_stripes"]
        allf = np.vstack([grids[r["image"]] for r in members])
        coreset = greedy_coreset(allf, k=min(CORESET_K, len(allf)), seed=42)
        bank = MemoryBank()
        bank.add(coreset)
        return bank

    report = {}
    for bank_name, include in [("with_stripes", True), ("without_stripes", False)]:
        bank = build_bank(include)
        local = {r["image"]: bank.patch_scores(grids[r["image"]]) for r in rows}
        for ctx_name, ctx, alphas in [("gradient", grad_ctx, GRAD_ALPHAS), ("fft", fft_ctx, FFT_ALPHAS)]:
            val_scores = {
                a: np.array([np.percentile(score_grid(local[r["image"]], ctx[r["image"]], alpha=a), PERCENTILE) for r in val])
                for a in alphas
            }
            val_auc = {a: float(roc_auc_score(val_labels, s)) for a, s in val_scores.items()}
            alpha_star = max(alphas, key=lambda a: (val_auc[a], -a))
            thr = youden_threshold(val_labels, val_scores[alpha_star])
            ts = np.array([np.percentile(score_grid(local[r["image"]], ctx[r["image"]], alpha=alpha_star), PERCENTILE) for r in test])
            m = operating_metrics(test_labels, ts, test_stripe, threshold=thr)
            m["auroc"] = float(roc_auc_score(test_labels, ts))
            key = f"{bank_name}_{ctx_name}"
            report[key] = {"val_auc": val_auc, "alpha_star": alpha_star, "test": m}
            print(f"{key} alpha*={alpha_star} test_auroc={m['auroc']:.4f} "
                  f"recall={m['faulty_recall']:.3f} stripe_fpr={m['stripe_fpr']:.2f}", flush=True)

    # bbox localization using with_stripes + gradient alpha*.
    bank = build_bank(True)
    local = {r["image"]: bank.patch_scores(grids[r["image"]]) for r in rows}
    alpha_loc = report["with_stripes_gradient"]["alpha_star"]
    with zipfile.ZipFile("data/raw/SADD_v1.1.zip") as archive:
        boxes = parse_labels_csv(io.StringIO(
            archive.read("SADD/labels/faulty/expert_annotations/labels.csv").decode("utf-8-sig")))
    hits = total = 0
    for r in test:
        if r["type"] != "faulty":
            continue
        grid = score_grid(local[r["image"]], grad_ctx[r["image"]], alpha=alpha_loc).reshape(14, 14)
        thr_map = np.percentile(grid, PERCENTILE)
        for box in boxes.get(r["image"], []):
            total += 1
            gx0, gy0, gx1, gy1 = box[0] / 1920 * 14, box[1] / 1050 * 14, box[2] / 1920 * 14, box[3] / 1050 * 14
            if bbox_hit(grid, (gx0, gy0, gx1, gy1), threshold=thr_map):
                hits += 1
    report["bbox_localization"] = {"hit_rate": hits / total if total else None, "hits": hits, "total_boxes": total}
    report["n_train"] = len(train)
    report["n_val"] = len(val)
    report["n_test"] = len(test)
    report["n_test_stripe"] = int(test_stripe.sum())
    report["coreset_k"] = CORESET_K
    with open(out / "region_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ROC for the main with_stripes_gradient configuration.
    ts = np.array([np.percentile(score_grid(local[r["image"]], grad_ctx[r["image"]], alpha=alpha_loc), PERCENTILE) for r in test])
    fpr, tpr, _ = roc_curve(test_labels, ts)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUROC={roc_auc_score(test_labels, ts):.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.legend(); plt.title("ROC (leakage-free region split)")
    plt.tight_layout()
    plt.savefig(out / "figure_roc.png", dpi=200)
    plt.savefig(out / "figure_roc.pdf")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
