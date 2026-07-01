import ast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from synthecg.config import SPLIT_FOLDS
from synthecg.data.ptbxl import select_records
from synthecg.export.ground_truth import build_annotation, save_signal
from synthecg.export.manifest import ManifestWriter
from synthecg.config import AugmentConfig, RenderConfig


def _make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["scp_codes"] = df["scp_codes"].apply(lambda x: x if isinstance(x, dict) else ast.literal_eval(x))
    df.index.name = "ecg_id"
    return df


def test_select_records_respects_split_and_seed():
    rows = []
    for ecg_id in range(1, 11):
        rows.append(
            {
                "ecg_id": ecg_id,
                "patient_id": ecg_id,
                "filename_hr": f"records500/00000/{ecg_id}_hr",
                "scp_codes": {"NORM": 80.0},
                "strat_fold": ecg_id,
            }
        )
    df = _make_df(rows).set_index("ecg_id")

    train = select_records(df, diagnosis="NORM", count=3, seed=42, split="train")
    assert set(train.strat_fold).issubset(SPLIT_FOLDS["train"])
    again = select_records(df, diagnosis="NORM", count=3, seed=42, split="train")
    assert list(train.index) == list(again.index)


def test_select_records_unique_patients():
    rows = [
        {"ecg_id": 1, "patient_id": 100, "filename_hr": "a", "scp_codes": {"NORM": 1}, "strat_fold": 1},
        {"ecg_id": 2, "patient_id": 100, "filename_hr": "b", "scp_codes": {"NORM": 1}, "strat_fold": 2},
        {"ecg_id": 3, "patient_id": 101, "filename_hr": "c", "scp_codes": {"NORM": 1}, "strat_fold": 3},
    ]
    df = _make_df(rows).set_index("ecg_id")
    sample = select_records(df, diagnosis="NORM", count=3, seed=1, unique_patients=True)
    assert sample["patient_id"].nunique() == len(sample)


def test_manifest_writer(tmp_path):
    writer = ManifestWriter(tmp_path)
    writer.add(
        sample_id="ecg_NORM_1",
        ecg_id=1,
        patient_id=10,
        diagnosis_query="NORM",
        scp_codes={"NORM": 80.0},
        strat_fold=1,
        image_path="images/ecg_NORM_1.png",
        signal_path="signals/ecg_NORM_1.npy",
        annotation_path="annotations/ecg_NORM_1.json",
        augmentations=["pink_tint"],
    )
    path = writer.write()
    text = path.read_text(encoding="utf-8")
    assert "ecg_NORM_1" in text
    assert "pink_tint" in text


def test_save_signal_and_annotation(tmp_path):
    record = MagicMock()
    record.p_signal = np.random.randn(5000, 12).astype(np.float32)
    record.fs = 500
    record.sig_len = 5000

    signal_path = save_signal(record, tmp_path / "test.npy")
    assert signal_path.exists()
    loaded = np.load(signal_path)
    assert loaded.shape == (12, 5000)

    annotation = build_annotation(
        ecg_id=1,
        patient_id=2,
        scp_codes={"NORM": 80.0},
        strat_fold=3,
        record=record,
        render=RenderConfig(),
        augment=AugmentConfig(profile="scan"),
        augmentations=["pink_tint"],
        image_path="images/test.png",
        signal_path="signals/test.npy",
    )
    assert annotation["signal"]["fs"] == 500
    assert annotation["render"]["layout"] == "3x4+1"


def test_select_records_unknown_diagnosis_raises():
    df = _make_df(
        [{"ecg_id": 1, "patient_id": 1, "filename_hr": "a", "scp_codes": {"NORM": 1}, "strat_fold": 1}]
    ).set_index("ecg_id")
    with pytest.raises(ValueError, match="No records found"):
        select_records(df, diagnosis="NOTACODE", count=1)
