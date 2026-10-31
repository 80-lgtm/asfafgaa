import csv
import io
import json
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.evaluation import operating_metrics, youden_threshold
from src.features import ResNet18Backend
from src.memory_bank import MemoryBank
from src.runner import feature_cache_key, score_grid, select_alpha
from src.sadd import archive_image_path
from src.stripe_context import stripe_consistency_map


def run_experiment(config, *, mode):
    output_dir = Path(config.output_dir) / mode
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config.data_dir) / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if mode == "smoke":
        rows = [row for row in rows if row["smoke"] == "1"]
    if not rows:
        raise ValueError(f"no rows selected for {mode}")
    backend = ResNet18Backend(device="cpu")
    backend.init_backend()
    features = {}
    timings = []
    with zipfile.ZipFile(config.archive_path) as archive:
        for index, row in enumerate(rows, start=1):
            raw = archive.read(archive_image_path(row))
            key = feature_cache_key(row["image"], raw, image_size=config.image_size)
            cache_path = cache_dir / key
            started = time.perf_counter()
            if cache_path.exists():
                with np.load(cache_path) as cached:
                    features[row["image"]] = {name: cached[name] for name in cached.files}
            else:
                image = np.asarray(Image.open(io.BytesIO(raw)).convert("L"))
                grid, shape = backend.extract_feature_grid(image, patch_size=config.image_size)
                context = stripe_consistency_map(image, grid_shape=shape)
                features[row["image"]] = {"grid": grid, "context": context.reshape(-1)}
                np.savez_compressed(cache_path, **features[row["image"]])
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
            row["seconds"] = elapsed
            print(f"[{mode}] {index}/{len(rows)} {row['image']} {row['type']} {elapsed:.3f}s", flush=True)
    bank = MemoryBank()
    for row in rows:
        if row["split"] == "train" and row["type"] != "faulty":
            bank.add(features[row["image"]]["grid"])
    # Precompute local scores once; alpha only rescales by stripe consistency.
    local_scores = {}
    contexts = {}
    for row in rows:
        item = features[row["image"]]
        local_scores[row["image"]] = bank.patch_scores(item["grid"])
        contexts[row["image"]] = item["context"]
    candidates = []
    for alpha in config.alpha_candidates:
        scored = {}
        for row in rows:
            adjusted = score_grid(local_scores[row["image"]], contexts[row["image"]], alpha=alpha)
            scored[row["image"]] = float(np.percentile(adjusted, config.image_score_percentile))
        val_rows = [row for row in rows if row["split"] == "val"]
        candidates.append(
            {
                "alpha": alpha,
                "val_labels": [int(row["type"] == "faulty") for row in val_rows],
                "val_scores": [scored[row["image"]] for row in val_rows],
                "scores": scored,
            }
        )
    best_alpha = select_alpha(candidates)
    selected = next(item for item in candidates if item["alpha"] == best_alpha)
    threshold = youden_threshold(selected["val_labels"], selected["val_scores"])
    test_rows = [row for row in rows if row["split"] == "test"]
    labels = np.array([int(row["type"] == "faulty") for row in test_rows])
    scores = np.array([selected["scores"][row["image"]] for row in test_rows])
    stripes = np.array([int(row["type"] == "good_stripes") for row in test_rows])
    metrics = operating_metrics(labels, scores, stripes, threshold=threshold)
    metrics["auroc"] = float(roc_auc_score(labels, scores))
    metrics["selected_alpha"] = float(best_alpha)
    metrics["n_images"] = len(rows)
    metrics["n_test"] = len(test_rows)
    metrics["mean_seconds_per_image"] = float(np.mean(timings))
    metrics["max_seconds_per_image"] = float(np.max(timings))
    metrics["memory_bank_vectors"] = int(bank.features.shape[0])
    metrics["memory_bank_dimensions"] = int(bank.features.shape[1])
    result_rows = []
    for row in rows:
        result = dict(row)
        result["score"] = selected["scores"][row["image"]]
        result["predicted_faulty"] = int(result["score"] >= threshold)
        result_rows.append(result)
    with (output_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    report = {
        "status": "completed",
        "mode": mode,
        "metrics": metrics,
        "alpha_validation": [
            {
                "alpha": item["alpha"],
                "val_auroc": float(roc_auc_score(item["val_labels"], item["val_scores"])),
            }
            for item in candidates
        ],
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "limitations": [
            "Small pilot; no significance or generalization claim.",
            "One time point per region, preserving official train/val/test splits.",
            "Bounding boxes are not used as pixel masks.",
            "Threshold and alpha are selected on validation data only.",
        ],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return report
