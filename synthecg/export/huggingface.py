"""Hugging Face dataset export and Hub publishing for SynthECG."""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np

HF_TOKEN_ENV = "HF_TOKEN"


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


def detect_layout(dataset_dir: str | Path) -> str:
    """Detect layout from the first annotation file, fallback to 3x4+1."""
    dataset_dir = Path(dataset_dir)
    annotations_dir = dataset_dir / "annotations"
    if not annotations_dir.exists():
        return "3x4+1"

    for path in sorted(annotations_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("render", {}).get("layout", "3x4+1")
    return "3x4+1"


def build_metadata_rows(dataset_dir: str | Path) -> list[dict]:
    """Build flat metadata records suitable for Parquet / Hugging Face datasets."""
    dataset_dir = Path(dataset_dir)
    rows = _load_manifest(dataset_dir)
    records: list[dict] = []

    for row in rows:
        sample_id = row["sample_id"]
        annotation_path = dataset_dir / "annotations" / f"{sample_id}.json"
        annotation = {}
        if annotation_path.exists():
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))

        signal_shape = None
        signal_path = dataset_dir / "signals" / f"{sample_id}.npy"
        if signal_path.exists():
            signal_shape = list(np.load(signal_path).shape)

        image_path = dataset_dir / row["image_path"]
        clean_path = row.get("clean_image_path", "")
        clean_abs = dataset_dir / clean_path if clean_path else None

        records.append(
            {
                "sample_id": sample_id,
                "ecg_id": int(row["ecg_id"]),
                "patient_id": int(row["patient_id"]),
                "diagnosis_query": row.get("diagnosis_query", ""),
                "scp_codes": row.get("scp_codes", "{}"),
                "strat_fold": int(row["strat_fold"]),
                "image": str(image_path.resolve()) if image_path.exists() else row["image_path"],
                "clean_image": str(clean_abs.resolve()) if clean_abs and clean_abs.exists() else "",
                "signal": str(signal_path.resolve()) if signal_path.exists() else "",
                "annotation": str(annotation_path.resolve()) if annotation_path.exists() else "",
                "mask": str((dataset_dir / row["mask_path"]).resolve()) if row.get("mask_path") else "",
                "yolo": str((dataset_dir / row["yolo_path"]).resolve()) if row.get("yolo_path") else "",
                "signal_shape": signal_shape,
                "layout": annotation.get("render", {}).get("layout", ""),
                "augment_profile": annotation.get("augment", {}).get("profile", ""),
                "augmentations": row.get("augmentations", "[]"),
            }
        )

    return records


def write_metadata_parquet(dataset_dir: str | Path, export_dir: str | Path) -> Path:
    """Write metadata.parquet for Hugging Face datasets integration."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Install HF support: pip install 'synthecg[hf]'") from exc

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    records = build_metadata_rows(dataset_dir)
    table = pa.Table.from_pylist(records)
    out_path = export_dir / "metadata.parquet"
    pq.write_table(table, out_path)
    return out_path


def build_dataset_card(
    dataset_dir: str | Path,
    *,
    layout: str = "3x4+1",
    repo_id: str | None = None,
) -> str:
    """Generate a Hugging Face dataset README/card from manifest statistics."""
    dataset_dir = Path(dataset_dir)
    rows = _load_manifest(dataset_dir)
    counts = _diagnosis_counts(rows)
    repo_line = f"repo_id: {repo_id}" if repo_id else ""

    size_bucket = 100
    for threshold in (100, 1000, 10000, 100000):
        if len(rows) < threshold:
            size_bucket = threshold
            break
    else:
        size_bucket = 1_000_000

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
        "  - digitization",
        "size_categories:",
        f"  - n<{size_bucket}",
    ]
    if repo_line:
        lines.append(repo_line)
    lines.extend(
        [
            "---",
            "",
            "# SynthECG Dataset",
            "",
            "Synthetic 12-lead ECG image dataset generated from [PTB-XL](https://physionet.org/content/ptb-xl/) using [SynthECG](https://github.com/umutinevi/EA-Syntetic-ECG-Creator).",
            "",
            "Each sample includes a rendered ECG image, ground-truth waveform, segmentation mask, YOLO labels, and JSON annotations with per-lead bounding boxes.",
            "",
            "## Dataset summary",
            "",
            f"- **Samples:** {len(rows)}",
            f"- **Layout:** `{layout}`",
            f"- **Source signals:** PTB-XL (PhysioNet)",
            "",
            "## Label distribution",
            "",
        ]
    )
    for label, count in sorted(counts.items()):
        lines.append(f"- `{label}`: {count}")

    lines.extend(
        [
            "",
            "## Load with Hugging Face Datasets",
            "",
            "```python",
            "from datasets import load_dataset",
            "",
            f'dataset = load_dataset("imagefolder", data_dir="images", metadata="metadata.parquet")',
            "print(dataset[0].keys())  # image, sample_id, ecg_id, diagnosis_query, ...",
            "```",
            "",
            "Or load from the Hub after publishing:",
            "",
            "```python",
            f'dataset = load_dataset("{repo_id or "your-username/synthecg-demo"}")',
            "```",
            "",
            "## Load signals in Python",
            "",
            "```python",
            "import numpy as np",
            "signal = np.load('signals/ecg_NORM_12345.npy')  # shape (12, n_samples), mV",
            "```",
            "",
            "## Directory layout",
            "",
            "| Path | Description |",
            "|------|-------------|",
            "| `images/` | Rendered ECG PNG |",
            "| `signals/` | Ground-truth `.npy` waveforms |",
            "| `annotations/` | JSON with lead bboxes |",
            "| `masks/` | Waveform segmentation masks |",
            "| `labels/` | YOLO bounding boxes |",
            "| `metadata.parquet` | Flat index for HF Datasets |",
            "",
            "## Citation",
            "",
            "Wagner, P., et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*.",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_hf_export(
    dataset_dir: str | Path,
    export_dir: str | Path | None = None,
    *,
    layout: str | None = None,
    repo_id: str | None = None,
) -> Path:
    """Prepare a Hugging Face-ready dataset folder with README, parquet, and assets."""
    dataset_dir = Path(dataset_dir)
    layout = layout or detect_layout(dataset_dir)
    export_dir = Path(export_dir) if export_dir else dataset_dir / "hf_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    for sub in ("images", "signals", "annotations", "masks", "labels"):
        src = dataset_dir / sub
        if src.exists():
            dst = export_dir / sub
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    clean_src = dataset_dir / "images" / "clean"
    if clean_src.exists():
        clean_dst = export_dir / "images" / "clean"
        if clean_dst.exists():
            shutil.rmtree(clean_dst)
        shutil.copytree(clean_src, clean_dst)

    manifest_src = dataset_dir / "manifest.csv"
    if manifest_src.exists():
        shutil.copy2(manifest_src, export_dir / "manifest.csv")

    write_metadata_parquet(dataset_dir, export_dir)

    card = build_dataset_card(dataset_dir, layout=layout, repo_id=repo_id)
    (export_dir / "README.md").write_text(card, encoding="utf-8")

    rows = _load_manifest(dataset_dir)
    meta = {
        "num_samples": len(rows),
        "layout": layout,
        "generator": "synthecg",
        "diagnosis_counts": dict(_diagnosis_counts(rows)),
        "repo_id": repo_id,
    }
    (export_dir / "dataset_info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return export_dir


def push_folder_to_hub(
    folder_path: str | Path,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
    commit_message: str = "Upload SynthECG dataset",
) -> str:
    """Upload a prepared folder to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("Install HF support: pip install 'synthecg[hf]'") from exc

    token = token or os.environ.get(HF_TOKEN_ENV)
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def push_as_hf_dataset(
    dataset_dir: str | Path,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
) -> str:
    """Push using Hugging Face `datasets` API with Image feature column."""
    try:
        from datasets import Dataset, Features, Image, Value
    except ImportError as exc:
        raise ImportError("Install HF support: pip install 'synthecg[hf]'") from exc

    token = token or os.environ.get(HF_TOKEN_ENV)
    records = build_metadata_rows(dataset_dir)

    features = Features(
        {
            "sample_id": Value("string"),
            "ecg_id": Value("int64"),
            "patient_id": Value("int64"),
            "diagnosis_query": Value("string"),
            "scp_codes": Value("string"),
            "strat_fold": Value("int64"),
            "image": Image(),
            "layout": Value("string"),
            "augment_profile": Value("string"),
        }
    )

    dataset = Dataset.from_list(
        [
            {
                "sample_id": r["sample_id"],
                "ecg_id": r["ecg_id"],
                "patient_id": r["patient_id"],
                "diagnosis_query": r["diagnosis_query"],
                "scp_codes": r["scp_codes"],
                "strat_fold": r["strat_fold"],
                "image": r["image"],
                "layout": r["layout"],
                "augment_profile": r["augment_profile"],
            }
            for r in records
        ],
        features=features,
    )
    dataset.push_to_hub(repo_id, private=private, token=token)
    return f"https://huggingface.co/datasets/{repo_id}"


def push_to_hub(
    dataset_dir: str | Path,
    repo_id: str,
    *,
    token: str | None = None,
    layout: str | None = None,
    export_dir: str | Path | None = None,
    private: bool = False,
    format: Literal["folder", "datasets"] = "folder",
) -> str:
    """Prepare and upload a SynthECG dataset to Hugging Face Hub."""
    if format == "datasets":
        return push_as_hf_dataset(dataset_dir, repo_id, token=token, private=private)

    prepared = prepare_hf_export(
        dataset_dir,
        export_dir=export_dir,
        layout=layout,
        repo_id=repo_id,
    )
    return push_folder_to_hub(prepared, repo_id, token=token, private=private)
