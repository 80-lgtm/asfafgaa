import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import zipfile

from src.pilot import stratified_sample
from src.sadd import median_time_per_region, parse_dataset_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a region-deduplicated 200-image SADD pilot")
    parser.add_argument("--archive", type=Path, default=Path("data/raw/SADD_v1.1.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    with zipfile.ZipFile(args.archive) as archive:
        rows = parse_dataset_csv(io.StringIO(archive.read("SADD/dataset.csv").decode("utf-8-sig")))
        rows = median_time_per_region(rows)
        # Normal categories have few independent regions, so use all of them.
        normal = [row for row in rows if row["type"] != "faulty"]
        faulty = stratified_sample(
            [row for row in rows if row["type"] == "faulty"],
            per_group=50,
            seed=args.seed,
        )
        selected = normal + faulty
        if len(selected) > 200:
            raise ValueError(f"pilot has {len(selected)} rows, expected at most 200")
        selected.sort(key=lambda row: (row["split"], row["type"], row["region"], row["image"]))
        for row in selected:
            row["smoke"] = "0"
        smoke_groups = Counter()
        for row in selected:
            key = (row["split"], row["type"])
            if smoke_groups[key] < 1:
                row["smoke"] = "1"
                smoke_groups[key] += 1
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
        summary = {
            "n_images": len(selected),
            "n_smoke": sum(row["smoke"] == "1" for row in selected),
            "counts": {str(key): value for key, value in Counter((row["split"], row["type"]) for row in selected).items()},
            "unique_regions": len(set(row["region"] for row in selected)),
            "source": "SADD v1.1.0, official train/val/test, median time per region",
        }
        with (args.output / "selection_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
