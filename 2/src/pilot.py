from collections import defaultdict
import random


def stratified_sample(rows, *, per_group, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["type"])].append(row)
    selected = []
    for key in sorted(groups):
        if len(groups[key]) < per_group:
            raise ValueError(f"insufficient rows for {key}: {len(groups[key])} < {per_group}")
        candidates = sorted(groups[key], key=lambda row: (row["region"], row["image"]))
        random.Random(f"{seed}:{key}").shuffle(candidates)
        selected.extend(candidates[:per_group])
    return selected
