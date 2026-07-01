from dataclasses import dataclass


@dataclass
class LeadRegion:
    name: str
    lead_idx: int
    bbox: tuple[int, int, int, int]
    label_bbox: tuple[int, int, int, int]
    plot_bbox: tuple[int, int, int, int]
    waveform_bbox: tuple[int, int, int, int]
    t_start: float
    t_end: float
    baseline_y: int


@dataclass
class RenderResult:
    image: object
    mask: object
    width: int
    height: int
    leads: list[LeadRegion]
    px_per_mm: float
    px_per_second: float
    px_per_mv: float
