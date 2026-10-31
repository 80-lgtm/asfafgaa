import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path
import zipfile

from src.sadd import median_time_per_region, parse_dataset_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the full region-deduplicated SADD manifest")
    parser.add_argument("--archive", type=Path, default=Path("data/raw/SADD_v1.1.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/processed_full"))
    args = parser.parse_args(argv)
    with zipfile.ZipFile(args.archive) as archive:
        rows = parse_dataset_csv(io.StringIO(archive.read("SADD/dataset.csv").decode("utf-8-sig")))
    rows = median_time_per_region(rows)
    rows.sort(key=lambda r: (r["split"], r["type"], r["region"], r["image"]))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "n_images": len(rows),
        "unique_regions": len(set(r["region"] for r in rows)),
        "counts": {str(k): v for k, v in Counter((r["split"], r["type"]) for r in rows).items()},
        "source": "SADD v1.1.0, official train/val/test, median time per region",
    }
    with (args.output / "selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
