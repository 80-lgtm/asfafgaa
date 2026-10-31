from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PilotConfig:
    data_dir: Path = Path("data/processed")
    output_dir: Path = Path("outputs")
    archive_path: Path = Path("data/raw/SADD_v1.1.zip")
    feature_backend: str = "resnet18"
    seed: int = 42
    image_size: int = 224
    patch_stride: int = 16
    alpha: float = 1.0
    alpha_candidates: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
    image_score_percentile: float = 95.0


def load_config(path: str | Path) -> PilotConfig:
    """Load a simple YAML pilot configuration without adding a schema dependency."""
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    allowed = PilotConfig.__dataclass_fields__.keys()
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if "data_dir" in values:
        values["data_dir"] = Path(values["data_dir"])
    if "output_dir" in values:
        values["output_dir"] = Path(values["output_dir"])
    if "archive_path" in values:
        values["archive_path"] = Path(values["archive_path"])
    if "alpha_candidates" in values:
        values["alpha_candidates"] = tuple(values["alpha_candidates"])
    return PilotConfig(**values)
