from pathlib import Path

from synthecg.config import YOLO_CLASSES
from synthecg.render.types import LeadRegion


def _to_yolo_line(class_id: int, bbox: tuple[int, int, int, int], img_w: int, img_h: int) -> str:
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def write_yolo_labels(
    leads: list[LeadRegion],
    output_path: str | Path,
    *,
    img_w: int,
    img_h: int,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for lead in leads:
        lines.append(_to_yolo_line(0, lead.bbox, img_w, img_h))
        lines.append(_to_yolo_line(1, lead.label_bbox, img_w, img_h))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_classes_file(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes_path = output_dir / "classes.txt"
    classes_path.write_text("\n".join(YOLO_CLASSES) + "\n", encoding="utf-8")
    return classes_path
