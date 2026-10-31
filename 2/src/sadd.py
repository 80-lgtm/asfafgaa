import csv
from collections import defaultdict
import statistics


def parse_dataset_csv(stream):
    """Parse the official semicolon-delimited SADD manifest."""
    return list(csv.DictReader(stream, delimiter=";"))


def archive_image_path(row):
    return f"SADD/images/{row['type']}/{row['split']}/{row['image']}"


def one_per_region(rows, *, time="1"):
    """Retain one time point per measurement region to limit near-duplicates."""
    if int(time) <= 0:
        raise ValueError("time must be positive")
    selected = {}
    for row in rows:
        if row["time"] != str(time):
            continue
        key = (row["region"], row["type"], row["split"])
        selected.setdefault(key, row)
    return list(selected.values())


def median_time_per_region(rows):
    """Select the middle frame of each region without assuming shared time ranges."""
    groups = defaultdict(list)
    for row in rows:
        key = (row["region"], row["type"], row["split"])
        groups[key].append(row)
    selected = []
    for key in sorted(groups):
        candidates = sorted(groups[key], key=lambda row: (int(row["time"]), row["image"]))
        times = [int(row["time"]) for row in candidates]
        target = statistics.median_low(times)
        selected.append(next(row for row in candidates if int(row["time"]) == target))
    return selected
