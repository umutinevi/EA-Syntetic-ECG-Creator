import ast
from unittest.mock import MagicMock

import cv2
import numpy as np
import pandas as pd
import pytest

from synthecg.config import SPLIT_FOLDS, AugmentConfig, RenderConfig
from synthecg.data.ptbxl import select_records
from synthecg.export.ground_truth import build_annotation, save_signal
from synthecg.export.manifest import ManifestWriter
from synthecg.export.yolo import write_yolo_labels
from synthecg.benchmark.digitize import pearson_correlation, run_benchmark
from synthecg.render.opencv_renderer import render_ecg_opencv
from synthecg.render.types import LeadRegion


def _make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["scp_codes"] = df["scp_codes"].apply(lambda x: x if isinstance(x, dict) else ast.literal_eval(x))
    df.index.name = "ecg_id"
    return df


def _make_record(fs=500, seconds=10):
    t = np.arange(int(fs * seconds)) / fs
    sine = np.sin(2 * np.pi * 1.25 * t)
    signals = np.stack([sine * (i + 1) * 0.05 for i in range(12)], axis=1)
    record = MagicMock()
    record.p_signal = signals.astype(np.float32)
    record.fs = fs
    record.sig_len = signals.shape[0]
    return record


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
        mask_path="masks/ecg_NORM_1.png",
        yolo_path="labels/ecg_NORM_1.txt",
        clean_image_path="images/clean/ecg_NORM_1.png",
        augmentations=["pink_tint"],
    )
    path = writer.write()
    text = path.read_text(encoding="utf-8")
    assert "ecg_NORM_1" in text
    assert "masks/ecg_NORM_1.png" in text


def test_save_signal_and_annotation(tmp_path):
    record = _make_record()
    signal_path = save_signal(record, tmp_path / "test.npy")
    assert signal_path.exists()
    loaded = np.load(signal_path)
    assert loaded.shape == (12, 5000)

    render_result = render_ecg_opencv(record, RenderConfig(), ecg_id=1, patient_id=2, scp_codes={"NORM": 80.0})
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
        render_result=render_result,
    )
    assert annotation["signal"]["fs"] == 500
    assert len(annotation["leads"]) == 13
    assert annotation["render"]["backend"] == "opencv"


def test_opencv_renderer_produces_image_and_mask():
    record = _make_record()
    result = render_ecg_opencv(record, RenderConfig(), ecg_id=99, patient_id=7, scp_codes={"NORM": 100.0})
    assert result.image.shape[0] == result.height
    assert result.mask.shape == (result.height, result.width)
    assert np.count_nonzero(result.mask) > 0
    assert len(result.leads) == 13


def test_yolo_label_format(tmp_path):
    leads = [
        LeadRegion(
            name="I",
            lead_idx=0,
            bbox=(10, 10, 100, 80),
            label_bbox=(20, 20, 30, 20),
            plot_bbox=(40, 30, 60, 50),
            waveform_bbox=(50, 30, 50, 50),
            t_start=0.0,
            t_end=2.5,
            baseline_y=60,
        )
    ]
    path = write_yolo_labels(leads, tmp_path / "sample.txt", img_w=200, img_h=100)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("0 ")
    assert lines[1].startswith("1 ")


def test_pearson_correlation_perfect():
    x = np.sin(np.linspace(0, 4 * np.pi, 200))
    assert pearson_correlation(x, x) == pytest.approx(1.0)


def test_select_records_unknown_diagnosis_raises():
    df = _make_df(
        [{"ecg_id": 1, "patient_id": 1, "filename_hr": "a", "scp_codes": {"NORM": 1}, "strat_fold": 1}]
    ).set_index("ecg_id")
    with pytest.raises(ValueError, match="No records found"):
        select_records(df, diagnosis="NOTACODE", count=1)
