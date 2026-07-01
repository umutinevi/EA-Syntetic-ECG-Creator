import csv
import json
from pathlib import Path


class ManifestWriter:
    """Append-only CSV manifest for generated dataset samples."""

    FIELDNAMES = [
        "sample_id",
        "ecg_id",
        "patient_id",
        "diagnosis_query",
        "scp_codes",
        "strat_fold",
        "image_path",
        "signal_path",
        "annotation_path",
        "mask_path",
        "yolo_path",
        "clean_image_path",
        "augmentations",
    ]

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.manifest_path = self.output_dir / "manifest.csv"
        self._rows: list[dict] = []

    def add(
        self,
        *,
        sample_id: str,
        ecg_id: int,
        patient_id: int,
        diagnosis_query: str,
        scp_codes: dict,
        strat_fold: int,
        image_path: str,
        signal_path: str | None,
        annotation_path: str | None,
        mask_path: str | None = None,
        yolo_path: str | None = None,
        clean_image_path: str | None = None,
        augmentations: list[str],
    ) -> None:
        self._rows.append(
            {
                "sample_id": sample_id,
                "ecg_id": ecg_id,
                "patient_id": patient_id,
                "diagnosis_query": diagnosis_query,
                "scp_codes": json.dumps(scp_codes),
                "strat_fold": strat_fold,
                "image_path": image_path,
                "signal_path": signal_path or "",
                "annotation_path": annotation_path or "",
                "mask_path": mask_path or "",
                "yolo_path": yolo_path or "",
                "clean_image_path": clean_image_path or "",
                "augmentations": json.dumps(augmentations),
            }
        )

    def write(self) -> Path:
        with self.manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(self._rows)
        return self.manifest_path
