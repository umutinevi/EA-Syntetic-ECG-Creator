"""Prepare and optionally publish SynthECG datasets to Hugging Face Hub."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path


def _load_manifest(dataset_dir: Path) -> list[dict]:
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _diagnosis_counts(rows: list[dict]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        counts[row.get("diagnosis_query", "unknown")] += 1
    return counts


def build_dataset_card(dataset_dir: str | Path, *, layout: str = "3x4+1") -> str:
    """Generate a Hugging Face dataset README/card from manifest statistics."""
    dataset_dir = Path(dataset_dir)
    rows = _load_manifest(dataset_dir)
    counts = _diagnosis_counts(rows)

    lines = [
        "---",
        "license: mit",
        "task_categories:",
        "  - image-to-image",
        "  - image-classification",
        "tags:",
        "  - medical",
        "  - ecg",
        "  - synthetic",
        "  - ptb-xl",
        f"size_categories:",
        f"  - n<{max(len(rows) * 2, 100)}",
        "---",
        "",
        "# SynthECG Dataset",
        "",
        f"Synthetic 12-lead ECG image dataset generated from PTB-XL using SynthECG.",
        "",
        "## Dataset summary",
        "",
        f"- **Samples:** {len(rows)}",
        f"- **Layout:** {layout}",
        f"- **Source signals:** PTB-XL (PhysioNet)",
        "",
        "## Label distribution",
        "",
    ]
    for label, count in sorted(counts.items()):
        lines.append(f"- `{label}`: {count}")
    lines.extend(
        [
            "",
            "## Files per sample",
            "",
            "- `images/` — rendered ECG PNG",
            "- `signals/` — ground-truth waveform `.npy` (12, n_samples)",
            "- `annotations/` — JSON with lead bboxes and render metadata",
            "- `masks/` — waveform segmentation masks",
            "- `labels/` — YOLO format bounding boxes",
            "",
            "## Citation",
            "",
            "Wagner, P., et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. Scientific Data.",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_hf_export(
    dataset_dir: str | Path,
    export_dir: str | Path | None = None,
    *,
    layout: str = "3x4+1",
) -> Path:
    """Prepare a Hugging Face-ready dataset folder with README and metadata."""
    dataset_dir = Path(dataset_dir)
    export_dir = Path(export_dir) if export_dir else dataset_dir / "hf_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    for sub in ("images", "signals", "annotations", "masks", "labels"):
        src = dataset_dir / sub
        if src.exists():
            dst = export_dir / sub
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    manifest_src = dataset_dir / "manifest.csv"
    if manifest_src.exists():
        shutil.copy2(manifest_src, export_dir / "manifest.csv")

    card = build_dataset_card(dataset_dir, layout=layout)
    (export_dir / "README.md").write_text(card, encoding="utf-8")

    rows = _load_manifest(dataset_dir)
    meta = {
        "num_samples": len(rows),
        "layout": layout,
        "generator": "synthecg",
        "diagnosis_counts": dict(_diagnosis_counts(rows)),
    }
    (export_dir / "dataset_info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return export_dir


def push_to_hub(
    dataset_dir: str | Path,
    repo_id: str,
    *,
    token: str | None = None,
    layout: str = "3x4+1",
    export_dir: str | Path | None = None,
) -> str:
    """Upload a prepared dataset folder to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("Install HF support: pip install 'synthecg[hf]'") from exc

    prepared = prepare_hf_export(dataset_dir, export_dir=export_dir, layout=layout)
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(prepared),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload SynthECG dataset",
    )
    return f"https://huggingface.co/datasets/{repo_id}"
