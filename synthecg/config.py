from dataclasses import dataclass, field
from typing import Literal


SPLIT_FOLDS = {
    "train": set(range(1, 9)),
    "val": {9},
    "test": {10},
}

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

YOLO_CLASSES = ["lead_region", "lead_label"]


@dataclass
class RenderConfig:
    layout: Literal["3x4+1", "12x1"] = "3x4+1"
    speed_mm_s: int = 25
    gain_mm_mv: int = 10
    dpi: int = 300
    backend: Literal["opencv", "matplotlib"] = "opencv"
    show_grid: bool = True
    grid_color: tuple[int, int, int] = (255, 180, 180)
    grid_minor_color: tuple[int, int, int] = (255, 220, 220)
    canvas_size: tuple[int, int] = (3508, 2480)


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
    workers: int = 1
    resume: bool = False
    include_codes: list[str] = field(default_factory=list)
    exclude_codes: list[str] = field(default_factory=list)
    balanced_codes: list[str] = field(default_factory=list)
    count_per_code: int | None = None
    bandpass_filter: bool = False
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    export_signals: bool = True
    export_annotations: bool = True
    export_masks: bool = True
    export_yolo: bool = True
    save_clean: bool = False
    render: RenderConfig = field(default_factory=RenderConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
