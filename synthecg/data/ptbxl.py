import ast
import csv
from pathlib import Path

import pandas as pd

from synthecg.config import SPLIT_FOLDS

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "synthecg"


def _cache_path(cache_dir: Path | str | None) -> Path:
    base = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / "ptbxl_database.csv"


def load_ptbxl_database(
    database: str = "ptb-xl/1.0.3",
    cache_dir: str | Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load PTB-XL metadata, caching the CSV locally after the first download."""
    cache_file = _cache_path(cache_dir)

    if cache_file.exists() and not force_refresh:
        print(f"Loading PTB-XL metadata from cache: {cache_file}")
        df = pd.read_csv(cache_file, index_col="ecg_id")
    else:
        url = f"https://physionet.org/files/{database}/ptbxl_database.csv"
        print(f"Downloading PTB-XL metadata from PhysioNet to {cache_file} ...")
        df = pd.read_csv(url, index_col="ecg_id")
        df.to_csv(cache_file)

    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    return df


def load_completed_ecg_ids(output_dir: str | Path) -> set[int]:
    """Load ecg_ids already present in an existing manifest."""
    manifest_path = Path(output_dir) / "manifest.csv"
    if not manifest_path.exists():
        return set()

    completed: set[int] = set()
    with manifest_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            completed.add(int(row["ecg_id"]))
    return completed


def _filter_by_diagnosis(df: pd.DataFrame, diagnosis: str) -> pd.DataFrame:
    if diagnosis.lower() == "random":
        return df
    filtered = df[df.scp_codes.apply(lambda codes: diagnosis in codes)]
    if filtered.empty:
        raise ValueError(f"No records found for diagnosis: {diagnosis}")
    return filtered


def _filter_by_codes(
    df: pd.DataFrame,
    include_codes: list[str] | None = None,
    exclude_codes: list[str] | None = None,
) -> pd.DataFrame:
    filtered = df
    if include_codes:
        include_set = set(include_codes)
        filtered = filtered[filtered.scp_codes.apply(lambda codes: include_set.issubset(codes.keys()))]
        if filtered.empty:
            raise ValueError(f"No records found including all codes: {include_codes}")
    if exclude_codes:
        exclude_set = set(exclude_codes)
        filtered = filtered[filtered.scp_codes.apply(lambda codes: exclude_set.isdisjoint(codes.keys()))]
        if filtered.empty:
            raise ValueError(f"No records remain after excluding codes: {exclude_codes}")
    return filtered


def _filter_by_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "all":
        return df
    if "strat_fold" not in df.columns:
        raise ValueError("PTB-XL metadata is missing strat_fold column required for split filtering.")
    allowed = SPLIT_FOLDS[split]
    return df[df.strat_fold.isin(allowed)]


def select_records(
    df: pd.DataFrame,
    diagnosis: str = "random",
    count: int = 5,
    seed: int | None = None,
    split: str = "all",
    unique_patients: bool = False,
    include_codes: list[str] | None = None,
    exclude_codes: list[str] | None = None,
    exclude_ecg_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Select PTB-XL records with optional split, seed, and patient de-duplication."""
    filtered = _filter_by_diagnosis(df, diagnosis)
    filtered = _filter_by_codes(filtered, include_codes, exclude_codes)
    filtered = _filter_by_split(filtered, split)

    if exclude_ecg_ids:
        filtered = filtered[~filtered.index.isin(exclude_ecg_ids)]

    if filtered.empty:
        raise ValueError(f"No records available for diagnosis='{diagnosis}' split='{split}'.")

    if unique_patients:
        filtered = filtered.drop_duplicates(subset="patient_id", keep="first")

    if count > len(filtered):
        print(f"Warning: Requested {count} records, but only {len(filtered)} available. Using all.")
        sample = filtered
    else:
        sample = filtered.sample(n=count, random_state=seed)

    return sample.sort_index()


def select_balanced_records(
    df: pd.DataFrame,
    codes: list[str],
    count_per_code: int,
    seed: int | None = None,
    split: str = "all",
    unique_patients: bool = False,
    exclude_ecg_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Select a balanced number of records per SCP code."""
    parts = []
    for index, code in enumerate(codes):
        code_seed = None if seed is None else seed + index * 1000
        sample = select_records(
            df,
            diagnosis=code,
            count=count_per_code,
            seed=code_seed,
            split=split,
            unique_patients=unique_patients,
            exclude_ecg_ids=exclude_ecg_ids,
        )
        sample = sample.copy()
        sample["diagnosis_query"] = code
        parts.append(sample)

    if not parts:
        raise ValueError("No balanced records selected.")

    return pd.concat(parts).sort_index()
