"""Ablation: good_stripes in/out of memory bank x gradient/FFT stripe context."""
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.evaluation import operating_metrics, youden_threshold
from src.memory_bank import MemoryBank, greedy_coreset
from src.runner import score_grid
from src.sadd import archive_image_path
from src.stripe_context import frequency_consistency_map, stripe_consistency_map

ALPHAS = [0.0, 1.0, 2.0, 4.0]
PERCENTILE = 95.0
CORESET_K = 2000


def load_grid(path):
    with np.load(path) as z:
        return z["grid"]


def main():
    out = Path("outputs/ablation")
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open("data/processed_full/manifest.csv", encoding="utf-8")))
    cache = Path("outputs/l2/cache")
    grids = {r["image"]: load_grid(cache / f"{r['image']}.npz") for r in rows}

    # Recompute frequency context from raw images; gradient context already cached.
    fft_cache_path = out / "fft_context.npz"
    grad_ctx = {r["image"]: np.load(cache / f"{r['image']}.npz")["context"] for r in rows}
    if fft_cache_path.exists():
        with np.load(fft_cache_path) as z:
            fft_ctx = {k: z[k] for k in z.files}
    else:
        fft_ctx = {}
        with zipfile.ZipFile("data/raw/SADD_v1.1.zip") as archive:
            for i, r in enumerate(rows, 1):
                raw = archive.read(archive_image_path(r))
                img = np.asarray(Image.open(io.BytesIO(raw)).convert("L"))
                fft_ctx[r["image"]] = frequency_consistency_map(img, grid_shape=(14, 14)).reshape(-1)
                if i % 250 == 0:
                    print(f"fft context {i}/{len(rows)}", flush=True)
        np.savez_compressed(fft_cache_path, **fft_ctx)

    train_clean = [r for r in rows if r["split"] == "train" and r["type"] == "good_clean"]
    train_stripes = [r for r in rows if r["split"] == "train" and r["type"] == "good_stripes"]
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    val_labels = np.array([int(r["type"] == "faulty") for r in val])
    test_labels = np.array([int(r["type"] == "faulty") for r in test])
    test_stripe = np.array([int(r["type"] == "good_stripes") for r in test])

    report = {}
    for bank_name, bank_rows in [
        ("bank_with_stripes", train_clean + train_stripes),
        ("bank_without_stripes", train_clean),
    ]:
        all_feats = np.vstack([grids[r["image"]] for r in bank_rows])
        coreset = greedy_coreset(all_feats, k=min(CORESET_K, len(all_feats)), seed=42)
        bank = MemoryBank()
        bank.add(coreset)
        local = {r["image"]: bank.patch_scores(grids[r["image"]]) for r in rows}
        for ctx_name, ctx in [("gradient", grad_ctx), ("fft", fft_ctx)]:
            val_scores = {
                a: np.array([
                    np.percentile(score_grid(local[r["image"]], ctx[r["image"]], alpha=a), PERCENTILE)
                    for r in val
                ])
                for a in ALPHAS
            }
            val_auc = {a: float(roc_auc_score(val_labels, s)) for a, s in val_scores.items()}
            alpha_star = max(ALPHAS, key=lambda a: (val_auc[a], -a))
            thr = youden_threshold(val_labels, val_scores[alpha_star])
            ts = np.array([
                np.percentile(score_grid(local[r["image"]], ctx[r["image"]], alpha=alpha_star), PERCENTILE)
                for r in test
            ])
            m = operating_metrics(test_labels, ts, test_stripe, threshold=thr)
            m["auroc"] = float(roc_auc_score(test_labels, ts))
            key = f"{bank_name}_{ctx_name}"
            report[key] = {"val_auc": val_auc, "alpha_star": alpha_star, "test": m}
            print(f"{key} alpha*={alpha_star} val_auc={val_auc[alpha_star]:.4f} "
                  f"test_auroc={m['auroc']:.4f} stripe_fpr={m['stripe_fpr']:.2f} "
                  f"recall={m['faulty_recall']:.3f}", flush=True)

    report["n_val"] = len(val)
    report["n_test"] = len(test)
    report["n_stripe_test"] = int(test_stripe.sum())
    report["coreset_k"] = CORESET_K
    with open(out / "ablation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
