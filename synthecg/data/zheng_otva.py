"""Zheng OT-VA dataset adapter (334 EP-validated outflow tract arrhythmias).

Data source: Zheng et al., Scientific Data 2020.
https://doi.org/10.1038/s41597-020-0440-8
"""

from __future__ import annotations

import csv
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from synthecg.config import LEAD_NAMES
from synthecg.localization.taxonomy import normalize_zheng_site, region_for_site, site_label
from synthecg.localization.types import LocalizationInfo

DATABASE_ID = "zheng-otva"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "synthecg" / "zheng_otva"
SOURCE_FS = 2000.0
TARGET_FS = 500.0

DIAGNOSIS_URL = "https://ndownloader.figshare.com/files/17675474"
ECG_ZIP_URL = "https://ndownloader.figshare.com/files/16838351"

ZHENG_LEAD_ALIASES = {
    "I": "I",
    "II": "II",
    "III": "III",
    "AVR": "aVR",
    "AVL": "aVL",
    "AVF": "aVF",
    "V1": "V1",
    "V2": "V2",
    "V3": "V3",
    "V4": "V4",
    "V5": "V5",
    "V6": "V6",
}


@dataclass
class ZhengPaths:
    root: Path
    diagnosis_csv: Path
    ecg_dir: Path


def _paths(cache_dir: Path | str | None) -> ZhengPaths:
    root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    return ZhengPaths(
        root=root,
        diagnosis_csv=root / "diagnosis.csv",
        ecg_dir=root / "ecg_denoised",
    )


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"Downloading {dest.name} from Figshare ...")
    urllib.request.urlretrieve(url, dest)


def _load_diagnosis_xlsx(xlsx_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(xlsx_path)
    except ImportError as exc:
        raise ImportError(
            "Reading Zheng diagnosis requires openpyxl. Install with: pip install openpyxl"
        ) from exc
    df = df.rename(columns=str.strip)
    return df


def ensure_zheng_data(cache_dir: Path | str | None = None, *, download_ecg: bool = True) -> ZhengPaths:
    """Download and extract Zheng OT-VA metadata and (optionally) ECG waveforms."""
    paths = _paths(cache_dir)
    paths.root.mkdir(parents=True, exist_ok=True)

    if not paths.diagnosis_csv.exists():
        xlsx_path = paths.root / "Diagnosis.xlsx"
        _download(DIAGNOSIS_URL, xlsx_path)
        df = _load_diagnosis_xlsx(xlsx_path)
        df.to_csv(paths.diagnosis_csv, index=False)
        print(f"Cached Zheng diagnosis -> {paths.diagnosis_csv}")

    if download_ecg and not paths.ecg_dir.exists():
        zip_path = paths.root / "PVCVTECGData.zip"
        _download(ECG_ZIP_URL, zip_path)
        extract_dir = paths.root / "_extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {zip_path.name} (this may take a minute) ...")
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)
        # Figshare zip may contain a nested folder.
        csv_files = list(extract_dir.rglob("*.csv"))
        if not csv_files:
            raise RuntimeError(f"No CSV files found in {zip_path}")
        paths.ecg_dir.mkdir(parents=True, exist_ok=True)
        for csv_file in csv_files:
            target = paths.ecg_dir / csv_file.name
            shutil.move(str(csv_file), target)
        shutil.rmtree(extract_dir, ignore_errors=True)
        print(f"Extracted {len(list(paths.ecg_dir.glob('*.csv')))} ECG CSV files -> {paths.ecg_dir}")

    return paths


def load_zheng_database(
    cache_dir: Path | str | None = None,
    *,
    download_ecg: bool = False,
) -> pd.DataFrame:
    """Load Zheng diagnostics indexed by hospital_id."""
    paths = ensure_zheng_data(cache_dir, download_ecg=download_ecg)
    df = pd.read_csv(paths.diagnosis_csv)
    df["hospital_id"] = df["HospitalID"].astype(int)
    df = df[df["Sublocation"].notna()]
    df["site"] = df["Sublocation"].apply(normalize_zheng_site)
    df["region"] = df["site"].apply(region_for_site)
    df["left_right"] = df["LeftRight"].str.strip()
    df = df.set_index("hospital_id")
    return df


def _find_ecg_csv(hospital_id: int, ecg_dir: Path) -> Path:
    candidates = [
        ecg_dir / f"{hospital_id}.csv",
        ecg_dir / f"{hospital_id}_hr.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(ecg_dir.glob(f"*{hospital_id}*.csv"))
    if len(matches) == 1:
        return matches[0]
    if matches:
        return sorted(matches)[0]
    raise FileNotFoundError(f"No ECG CSV found for hospital_id={hospital_id} in {ecg_dir}")


def _read_ecg_csv(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    columns = {col: col.strip().upper() for col in df.columns}
    ordered = []
    for lead in LEAD_NAMES:
        match = None
        for raw, normalized in columns.items():
            alias = ZHENG_LEAD_ALIASES.get(normalized, normalized)
            if alias == lead:
                match = raw
                break
        if match is None:
            raise ValueError(f"Lead {lead} missing in {csv_path.name}; columns={list(df.columns)}")
        ordered.append(df[match].to_numpy(dtype=np.float32))
    signal = np.stack(ordered, axis=1)
    return signal


def _resample_signal(signal: np.ndarray, source_fs: float, target_fs: float) -> np.ndarray:
    if abs(source_fs - target_fs) < 1e-3:
        return signal.astype(np.float32)
    ratio = target_fs / source_fs
    # resample_poly expects up/down integers; use 1/4 for 2000->500.
    if abs(source_fs / target_fs - 4.0) < 1e-3:
        return resample_poly(signal, 1, 4, axis=0).astype(np.float32)
    n_target = int(round(signal.shape[0] * ratio))
    t_source = np.linspace(0, 1, signal.shape[0], endpoint=False)
    t_target = np.linspace(0, 1, n_target, endpoint=False)
    resampled = np.stack(
        [np.interp(t_target, t_source, signal[:, i]) for i in range(signal.shape[1])],
        axis=1,
    )
    return resampled.astype(np.float32)


def localization_from_row(row) -> LocalizationInfo:
    site = row.site if hasattr(row, "site") else normalize_zheng_site(row["Sublocation"])
    region = row.region if hasattr(row, "region") else region_for_site(site)
    return LocalizationInfo.from_ep(region=region or "RVOT", site=site)


def fetch_zheng_record(
    hospital_id: int,
    cache_dir: Path | str | None = None,
) -> SimpleNamespace:
    """Load one Zheng 12-lead ECG record resampled to 500 Hz."""
    paths = ensure_zheng_data(cache_dir, download_ecg=True)
    csv_path = _find_ecg_csv(hospital_id, paths.ecg_dir)
    print(f"Loading Zheng record hospital_id={hospital_id} from {csv_path.name} ...")
    signal = _read_ecg_csv(csv_path)
    signal = _resample_signal(signal, SOURCE_FS, TARGET_FS)
    return SimpleNamespace(
        p_signal=signal,
        fs=TARGET_FS,
        sig_len=signal.shape[0],
        sig_name=str(hospital_id),
        units="mV",
    )


def select_zheng_records(
    df: pd.DataFrame,
    *,
    count: int = 5,
    seed: int | None = None,
    site: str | None = None,
    region: str | None = None,
    balanced_sites: list[str] | None = None,
    count_per_site: int | None = None,
    exclude_hospital_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Select Zheng records with optional site/region filters."""
    filtered = df.copy()
    if exclude_hospital_ids:
        filtered = filtered[~filtered.index.isin(exclude_hospital_ids)]

    if balanced_sites:
        per_site = count_per_site or max(1, count // len(balanced_sites))
        parts = []
        for index, raw_site in enumerate(balanced_sites):
            from synthecg.localization.taxonomy import ZHENG_SITE_MAP

            canonical = ZHENG_SITE_MAP.get(raw_site, raw_site)
            reverse = {value: key for key, value in ZHENG_SITE_MAP.items()}
            raw_label = reverse.get(raw_site, raw_site)
            site_df = filtered[
                (filtered["Sublocation"] == raw_site)
                | (filtered["Sublocation"] == raw_label)
                | (filtered["site"] == raw_site)
                | (filtered["site"] == canonical)
            ]
            if site_df.empty:
                raise ValueError(f"No Zheng records for site={raw_site}")
            site_seed = None if seed is None else seed + index * 1000
            sample = site_df.sample(n=min(per_site, len(site_df)), random_state=site_seed)
            sample = sample.copy()
            sample["diagnosis_query"] = raw_site
            parts.append(sample)
        return pd.concat(parts).sort_index()

    if site:
        canonical = normalize_zheng_site(site) if site not in df["site"].values else site
        filtered = filtered[(filtered["Sublocation"] == site) | (filtered["site"] == site) | (filtered["site"] == canonical)]
    if region:
        filtered = filtered[filtered["region"].str.upper() == region.upper()]

    if filtered.empty:
        raise ValueError(f"No Zheng records available for site={site!r} region={region!r}")

    if count > len(filtered):
        print(f"Warning: requested {count} Zheng records, using all {len(filtered)}.")
        sample = filtered
    else:
        sample = filtered.sample(n=count, random_state=seed)
    return sample.sort_index()


def load_completed_hospital_ids(output_dir: str | Path) -> set[int]:
    manifest_path = Path(output_dir) / "manifest.csv"
    if not manifest_path.exists():
        return set()
    completed: set[int] = set()
    with manifest_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            completed.add(int(row["ecg_id"]))
    return completed


def zheng_metadata_row(row) -> dict:
    loc = localization_from_row(row)
    return {
        "ecg_id": int(row.name),
        "patient_id": int(row.name),
        "hospital_id": int(row.name),
        "filename_hr": str(row.name),
        "scp_codes": {"PVC": 100.0},
        "strat_fold": 0,
        "site": loc.site,
        "region": loc.region,
        "diagnosis_query": getattr(row, "diagnosis_query", row.site),
        "localization": loc.to_dict(),
        "site_label": site_label(loc.site or ""),
    }
