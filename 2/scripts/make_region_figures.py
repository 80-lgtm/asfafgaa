"""Traceable result figures from the verified leakage-free region split."""
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

from src.localization import parse_labels_csv
from src.memory_bank import MemoryBank, greedy_coreset
from src.runner import score_grid
from src.sadd import archive_image_path

CORESET_K = 2000
PERCENTILE = 95.0
CONFIGS = [
    ("with_stripes_gradient", "C+S / G"),
    ("with_stripes_fft", "C+S / F"),
    ("without_stripes_gradient", "C / G"),
    ("without_stripes_fft", "C / F"),
]


def main():
    out = Path("outputs/region")
    report = json.load(open(out / "region_report.json", encoding="utf-8"))

    names = [label for _, label in CONFIGS]
    val_auc = [report[k]["val_auc"][str(report[k]["alpha_star"])] for k, _ in CONFIGS]
    test_auc = [report[k]["test"]["auroc"] for k, _ in CONFIGS]
    x = np.arange(len(CONFIGS))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, val_auc, 0.4, label="validation AUROC")
    ax.bar(x + 0.2, test_auc, 0.4, label="test AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("AUROC")
    ax.legend(fontsize=8)
    ax.set_title("Normal-mode diversity ablation (region-disjoint splits)")
    fig.tight_layout()
    fig.savefig(out / "figure_auc_bars.png", dpi=200)
    fig.savefig(out / "figure_auc_bars.pdf")
    plt.close(fig)

    stripe_fpr = [report[k]["test"]["stripe_fpr"] for k, _ in CONFIGS]
    normal_fpr = [report[k]["test"]["normal_fpr"] for k, _ in CONFIGS]
    rows = list(csv.DictReader(open("data/processed_region/manifest.csv", encoding="utf-8")))
    test_normals = [r for r in rows if r["split"] == "test" and r["type"] != "faulty"]
    n_stripe = sum(r["type"] == "good_stripes" for r in test_normals)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, stripe_fpr, 0.4, label=f"Good_stripes FPR (n={n_stripe} images)")
    ax.bar(x + 0.2, normal_fpr, 0.4, label=f"all-normal FPR (n={len(test_normals)} images)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("False positive rate")
    ax.legend(fontsize=8)
    ax.set_title("False positives on normal test samples")
    fig.tight_layout()
    fig.savefig(out / "figure_stripe_fpr.png", dpi=200)
    fig.savefig(out / "figure_stripe_fpr.pdf")
    plt.close(fig)

    # One faulty test image: grayscale, anomaly map, bbox overlay.
    feat_cache = Path("outputs/l2/cache")
    train_normals = [r for r in rows if r["split"] == "train" and r["type"] != "faulty"]
    grids = {r["image"]: np.load(feat_cache / f"{r['image']}.npz")["grid"] for r in rows}
    allf = np.vstack([grids[r["image"]] for r in train_normals])
    bank = MemoryBank()
    bank.add(greedy_coreset(allf, k=min(CORESET_K, len(allf)), seed=42))
    alpha = report["with_stripes_gradient"]["alpha_star"]
    target = next(r for r in rows if r["split"] == "test" and r["type"] == "faulty")
    with np.load(feat_cache / f"{target['image']}.npz") as z:
        ctx = z["context"]
    heat = score_grid(bank.patch_scores(grids[target["image"]]), ctx, alpha=alpha).reshape(14, 14)
    with zipfile.ZipFile("data/raw/SADD_v1.1.zip") as archive:
        image = np.asarray(Image.open(io.BytesIO(archive.read(archive_image_path(target)))).convert("L"))
        boxes = parse_labels_csv(io.StringIO(
            archive.read("SADD/labels/faulty/expert_annotations/labels.csv").decode("utf-8-sig")))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input (faulty)")
    axes[1].imshow(heat, cmap="viridis", extent=[0, image.shape[1], image.shape[0], 0])
    axes[1].set_title("Anomaly map (14x14, upscaled view)")
    axes[2].imshow(image, cmap="gray")
    for xmin, ymin, xmax, ymax in boxes.get(target["image"], []):
        axes[2].add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        fill=False, edgecolor="red", lw=1.5))
    axes[2].set_title("Expert bounding boxes")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "figure_heatmap_example.png", dpi=200)
    plt.close(fig)
    print("figures written to", out)


if __name__ == "__main__":
    main()
