import csv
from collections import defaultdict
import random
from pathlib import Path


REQUIRED_COLUMNS = {"image_path", "label", "split"}


def validate_manifest(rows) -> list[dict]:
    """Validate manifest rows before any image is loaded."""
    if not rows:
        raise ValueError("manifest is empty")
    if isinstance(rows, dict):
        columns = set(rows["columns"]) if "columns" in rows else set(rows)
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
        rows = rows.get("rows", [])
    elif rows:
        missing = REQUIRED_COLUMNS - set(rows[0])
        if missing:
            raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
    else:
        raise ValueError("manifest is empty")
    return [dict(row) for row in rows]


def deterministic_split(rows, *, seed: int, train: int, validation: int, test: int):
    """Create a reproducible split without mutating the input rows."""
    if train + validation + test != len(rows):
        raise ValueError("split sizes must sum to the number of rows")
    ordered = list(rows)
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    return {
        "train": shuffled[:train],
        "validation": shuffled[train : train + validation],
        "test": shuffled[train + validation : train + validation + test],
    }


def manifest_to_csv(manifest_path: str | Path, output_path: str | Path):
    rows = validate_manifest(list(csv.DictReader(Path(manifest_path).open("r", encoding="utf-8"))))
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
