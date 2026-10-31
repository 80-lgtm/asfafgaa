"""Build a leakage-free manifest where each physical region belongs to exactly
one split. Faulty regions keep their official split. Normal regions that span
multiple splits are assigned to a single split with priority test > val > train
to preserve scarce evaluation coverage; ties are broken by region id for
determinism."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}


def assign_split(splits, region):
    if len(splits) == 1:
        return next(iter(splits))
    return min(splits, key=lambda s: (SPLIT_PRIORITY[s], region))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed_full/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed_region"))
    args = parser.parse_args(argv)
    rows = list(csv.DictReader(open(args.input, encoding="utf-8")))
    groups = defaultdict(list)
    for r in rows:
        groups[r["region"]].append(r)
    for region, members in groups.items():
        splits = {m["split"] for m in members}
        target = assign_split(splits, region)
        for m in members:
            m["split"] = target
    corrected = [m for members in groups.values() for m in members]
    corrected.sort(key=lambda r: (r["split"], r["type"], r["region"], r["image"]))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(corrected[0]))
        w.writeheader()
        w.writerows(corrected)
    summary = {
        "n_images": len(corrected),
        "unique_regions": len(set(r["region"] for r in corrected)),
        "counts": {str(k): v for k, v in Counter((r["split"], r["type"]) for r in corrected).items()},
        "train_test_region_overlap": _overlap(corrected, "train", "test"),
        "train_val_region_overlap": _overlap(corrected, "train", "val"),
        "val_test_region_overlap": _overlap(corrected, "val", "test"),
    }
    with (args.output / "region_split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


def _overlap(rows, a, b):
    reg_a = {r["region"] for r in rows if r["split"] == a}
    reg_b = {r["region"] for r in rows if r["split"] == b}
    return len(reg_a & reg_b)


if __name__ == "__main__":
    raise SystemExit(main())
