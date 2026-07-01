from dataclasses import dataclass, field
from typing import Literal


SPLIT_FOLDS = {
    "train": set(range(1, 9)),
    "val": {9},
    "test": {10},
}

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@dataclass
class RenderConfig:
    layout: str = "3x4+1"
    speed_mm_s: int = 25
    gain_mm_mv: int = 10
    dpi: int = 150
    figsize: tuple[float, float] = (11.0, 8.0)


@dataclass
class AugmentConfig:
    profile: Literal["clean", "scan", "clinical"] = "scan"
    seed: int | None = None


@dataclass
class GenerationConfig:
    count: int = 5
    diagnosis: str = "random"
    output_dir: str = "output_ecgs"
    database: str = "ptb-xl/1.0.3"
    cache_dir: str | None = None
    seed: int | None = None
    split: Literal["all", "train", "val", "test"] = "all"
    unique_patients: bool = False
    export_signals: bool = True
    export_annotations: bool = True
    render: RenderConfig = field(default_factory=RenderConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
