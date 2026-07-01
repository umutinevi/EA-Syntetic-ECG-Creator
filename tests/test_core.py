import ast
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from synthecg.config import SPLIT_FOLDS, AugmentConfig, RenderConfig
from synthecg.data.preprocess import bandpass_filter_signal, preprocess_record
from synthecg.data.ptbxl import load_completed_ecg_ids, select_balanced_records, select_records
from synthecg.export.ground_truth import build_annotation, save_signal
from synthecg.export.manifest import ManifestWriter
from synthecg.export.yolo import write_yolo_labels
from synthecg.benchmark.digitize import pearson_correlation
from synthecg.recipes.builder import config_from_recipe
from synthecg.recipes.definitions import get_recipe, list_recipes
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


def test_select_records_include_exclude_codes():
    rows = [
        {"ecg_id": 1, "patient_id": 1, "filename_hr": "a", "scp_codes": {"NORM": 1, "SR": 1}, "strat_fold": 1},
        {"ecg_id": 2, "patient_id": 2, "filename_hr": "b", "scp_codes": {"NORM": 1, "AFIB": 1}, "strat_fold": 2},
    ]
    df = _make_df(rows).set_index("ecg_id")
    sample = select_records(df, diagnosis="random", count=5, include_codes=["NORM", "SR"])
    assert len(sample) == 1
    assert sample.index[0] == 1

    sample = select_records(df, diagnosis="random", count=5, exclude_codes=["AFIB"])
    assert len(sample) == 1


def test_select_balanced_records():
    rows = []
    for ecg_id, code in enumerate(["NORM", "NORM", "AFIB", "AFIB"], start=1):
        rows.append(
            {
                "ecg_id": ecg_id,
                "patient_id": ecg_id,
                "filename_hr": f"f{ecg_id}",
                "scp_codes": {code: 1.0},
                "strat_fold": 1,
            }
        )
    df = _make_df(rows).set_index("ecg_id")
    sample = select_balanced_records(df, codes=["NORM", "AFIB"], count_per_code=1, seed=1)
    assert len(sample) == 2
    assert set(sample["diagnosis_query"]) == {"NORM", "AFIB"}


def test_manifest_writer_resume(tmp_path):
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
        augmentations=[],
    )
    writer.write()

    resumed = ManifestWriter(tmp_path, resume=True)
    assert len(resumed._rows) == 1
    assert load_completed_ecg_ids(tmp_path) == {1}


def test_bandpass_filter():
    record = _make_record()
    filtered = preprocess_record(record, bandpass=True)
    assert filtered.p_signal.shape == record.p_signal.shape
    assert not np.allclose(filtered.p_signal, record.p_signal)


def test_recipe_config():
    config = config_from_recipe("clean-baseline", output_dir="/tmp/test", seed=1)
    assert config.augment.profile == "clean"
    assert config.bandpass_filter is True
    assert config.save_clean is True
    assert "clean-baseline" in list_recipes()
    assert get_recipe("clean-baseline")["count"] == 20


def test_opencv_renderer_no_grid():
    record = _make_record()
    result = render_ecg_opencv(
        record,
        RenderConfig(show_grid=False),
        ecg_id=1,
        patient_id=1,
        scp_codes={"NORM": 1},
    )
    assert result.image is not None


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


def test_pearson_correlation_perfect():
    x = np.sin(np.linspace(0, 4 * np.pi, 200))
    assert pearson_correlation(x, x) == pytest.approx(1.0)


def test_select_records_unknown_diagnosis_raises():
    df = _make_df(
        [{"ecg_id": 1, "patient_id": 1, "filename_hr": "a", "scp_codes": {"NORM": 1}, "strat_fold": 1}]
    ).set_index("ecg_id")
    with pytest.raises(ValueError, match="No records found"):
        select_records(df, diagnosis="NOTACODE", count=1)
