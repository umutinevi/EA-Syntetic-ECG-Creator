"""Simple column-scan ECG digitizer for round-trip benchmark evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _extract_trace_y(column: np.ndarray, baseline_rel: int, search_half_height: int = 80) -> int | None:
    y0 = max(0, baseline_rel - search_half_height)
    y1 = min(len(column), baseline_rel + search_half_height)
    band = column[y0:y1]
    if band.size == 0:
        return None

    darkest_rel = int(np.argmin(band))
    darkest_val = band[darkest_rel]
    if darkest_val > 120:
        return None
    return y0 + darkest_rel


def digitize_lead(
    gray: np.ndarray,
    waveform_bbox: tuple[int, int, int, int],
    baseline_y: int,
    px_per_mv: float,
    n_samples: int,
) -> np.ndarray:
    """Digitize one lead region by column-wise trace tracking."""
    x, y, w, h = waveform_bbox
    crop = gray[y : y + h, x : x + w]
    if crop.size == 0 or w < 2:
        return np.zeros(n_samples, dtype=np.float32)

    signal = np.zeros(n_samples, dtype=np.float32)
    baseline_rel = baseline_y - y
    cols = np.linspace(0, max(w - 1, 1), n_samples).astype(int)

    for i, col_idx in enumerate(cols):
        column = crop[:, col_idx]
        trace_y = _extract_trace_y(column, baseline_rel)
        if trace_y is not None:
            signal[i] = (baseline_y - (y + trace_y)) / px_per_mv

    return signal


def digitize_sample(
    image_path: str | Path,
    annotation_path: str | Path,
) -> dict[str, np.ndarray]:
    """Digitize all annotated leads from one ECG image."""
    annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    if "leads" not in annotation:
        raise ValueError(f"Annotation missing lead layout metadata: {annotation_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    px_per_mv = annotation["render"]["px_per_mv"]
    fs = annotation["signal"]["fs"]
    digitized: dict[str, np.ndarray] = {}

    for lead in annotation["leads"]:
        duration = lead["t_end"] - lead["t_start"]
        n_samples = max(2, int(round(duration * fs)))
        bbox = tuple(lead.get("waveform_bbox", lead["plot_bbox"]))
        trace = digitize_lead(
            img,
            bbox,
            lead["baseline_y"],
            px_per_mv,
            n_samples,
        )
        digitized[lead["name"] + f"_{lead['t_start']}-{lead['t_end']}"] = trace

    return digitized


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_sample(
    dataset_dir: str | Path,
    sample_id: str,
    *,
    use_clean: bool = True,
) -> dict:
    """Compare digitized traces against ground-truth PTB-XL signal segments."""
    dataset_dir = Path(dataset_dir)
    annotation_path = dataset_dir / "annotations" / f"{sample_id}.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))

    if use_clean and annotation["paths"].get("clean_image"):
        image_path = dataset_dir / annotation["paths"]["clean_image"]
    else:
        image_path = dataset_dir / annotation["paths"]["image"]

    signal_path = dataset_dir / annotation["paths"]["signal"]
    ground_truth = np.load(signal_path)
    fs = annotation["signal"]["fs"]

    digitized = digitize_sample(image_path, annotation_path)
    lead_scores: dict[str, float] = {}

    for lead in annotation["leads"]:
        key = lead["name"] + f"_{lead['t_start']}-{lead['t_end']}"
        if key not in digitized:
            continue

        idx_start = int(lead["t_start"] * fs)
        idx_end = int(lead["t_end"] * fs)
        gt = ground_truth[lead["lead_idx"], idx_start:idx_end]
        pred = digitize_lead(
            cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
            tuple(lead.get("waveform_bbox", lead["plot_bbox"])),
            lead["baseline_y"],
            annotation["render"]["px_per_mv"],
            len(gt),
        )
        lead_scores[key] = pearson_correlation(gt, pred)

    valid = [v for v in lead_scores.values() if not np.isnan(v)]
    mean_corr = float(np.mean(valid)) if valid else float("nan")

    return {
        "sample_id": sample_id,
        "image_used": str(image_path.relative_to(dataset_dir)),
        "lead_correlations": lead_scores,
        "mean_correlation": mean_corr,
        "n_leads_scored": len(valid),
    }


def run_benchmark(
    dataset_dir: str | Path,
    *,
    use_clean: bool = True,
    limit: int | None = None,
) -> dict:
    """Run round-trip digitization benchmark on a generated dataset."""
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    import csv

    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    if limit is not None:
        rows = rows[:limit]

    results = []
    for row in rows:
        try:
            results.append(evaluate_sample(dataset_dir, row["sample_id"], use_clean=use_clean))
        except Exception as exc:
            results.append({"sample_id": row["sample_id"], "error": str(exc), "mean_correlation": float("nan")})

    valid = [r["mean_correlation"] for r in results if "mean_correlation" in r and not np.isnan(r["mean_correlation"])]
    summary = {
        "dataset_dir": str(dataset_dir),
        "use_clean": use_clean,
        "n_samples": len(results),
        "mean_correlation": float(np.mean(valid)) if valid else float("nan"),
        "results": results,
    }

    report_path = dataset_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
