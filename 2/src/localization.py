import csv

import numpy as np


def parse_labels_csv(stream):
    """Parse SADD expert labels into {image: [(xmin, ymin, xmax, ymax), ...]}."""
    boxes = {}
    for row in csv.DictReader(stream, delimiter=";"):
        box = (int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"]))
        boxes.setdefault(row["image"], []).append(box)
    return boxes


def bbox_hit(heatmap: np.ndarray, box, *, threshold: float) -> bool:
    """A bbox counts as hit if any pixel above threshold falls inside it."""
    heatmap = np.asarray(heatmap)
    xmin, ymin, xmax, ymax = [int(v) for v in box]
    h, w = heatmap.shape
    xmin, ymin = max(xmin, 0), max(ymin, 0)
    xmax, ymax = min(xmax, w), min(ymax, h)
    if xmax <= xmin or ymax <= ymin:
        return False
    return bool(np.any(heatmap[ymin:ymax, xmin:xmax] >= threshold))
