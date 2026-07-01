"""Tests for arrhythmia localization taxonomy, inference, and Zheng adapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from synthecg.config import is_zheng_database
from synthecg.data.zheng_otva import (
    fetch_zheng_record,
    load_zheng_database,
    localization_from_row,
    select_zheng_records,
)
from synthecg.export.ground_truth import build_annotation
from synthecg.localization.inference import infer_localization_from_signal
from synthecg.localization.pvc_otva import infer_pvc_localization
from synthecg.localization.taxonomy import normalize_zheng_site, region_for_site, site_label
from synthecg.localization.wpw import infer_wpw_localization
from synthecg.recipes.builder import config_from_recipe

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "zheng"


def _synthetic_ot_pvc_signal(fs: float = 500.0, seconds: float = 4.0) -> np.ndarray:
    """Build a synthetic 12-lead signal with one ectopic beat."""
    n = int(fs * seconds)
    t = np.arange(n) / fs
    signal = np.zeros((12, n), dtype=np.float32)
    for lead in range(12):
        signal[lead] = 0.05 * np.sin(2 * np.pi * 1.0 * t)

    pvc_center = int(2.0 * fs)
    width = int(0.08 * fs)
    for lead in range(12):
        signal[lead, pvc_center - width : pvc_center + width] += 0.15 * (lead + 1)

    # LVOT-like morphology: boost V2 ectopic and add small r in V1
    signal[7, pvc_center - width : pvc_center + width] += 0.9  # V2
    signal[6, pvc_center - width : pvc_center + width] += 0.08  # V1 r hint
    return signal


def test_taxonomy_normalization():
    assert normalize_zheng_site("LCC") == "LCC"
    assert normalize_zheng_site("FreeWall") == "free_wall"
    assert region_for_site("LCC") == "LVOT"
    assert region_for_site("free_wall") == "RVOT"
    assert site_label("LCC") == "Left Coronary Cusp"


def test_is_zheng_database():
    assert is_zheng_database("zheng-otva")
    assert is_zheng_database("Zheng")
    assert not is_zheng_database("ptb-xl/1.0.3")


def test_pvc_algorithm_lvot_lcc():
    signal = _synthetic_ot_pvc_signal()
    result = infer_pvc_localization(signal, fs=500.0)
    assert result is not None
    assert result.region == "LVOT"
    assert result.source == "algorithm"
    assert result.verified is False
    assert result.confidence <= 0.75
    assert result.algorithm == "pvc_otva_literature_v2"
    assert result.features.get("v2_transition_ratio", 0) >= 0.6


def test_wpw_literature_rules_cs_os_pattern():
    """I− / II+ / V1− / low R/S should map to CS OS, not free wall (Milstein + combined rule)."""
    fs = 500.0
    n = 100
    qrs = np.zeros((12, n), dtype=np.float32)
    qrs[6] = -0.2  # V1 negative dominant

    with patch("synthecg.localization.wpw.extract_delta_polarities") as mock_pol:
        mock_pol.return_value = {"I": "neg", "II": "pos", "III": "iso", "aVF": "pos", "V1": "neg"}
        with patch("synthecg.localization.wpw.extract_qrs_segment_at_peak", return_value=qrs):
            result = infer_wpw_localization(np.zeros((12, 500), dtype=np.float32), fs)
    assert result is not None
    assert result.site == "coronary_sinus_ostium"
    assert result.site != "right_free_wall"
    assert result.verified is False
    assert result.confidence <= 0.50


def test_wpw_arruda_negative_ii_coronary_venous():
    fs = 500
    n = 100
    qrs = np.zeros((12, n), dtype=np.float32)
    with patch("synthecg.localization.wpw.extract_delta_polarities") as mock_pol:
        mock_pol.return_value = {"I": "neg", "II": "neg", "III": "neg", "aVF": "neg", "V1": "pos"}
        with patch("synthecg.localization.wpw.extract_qrs_segment_at_peak", return_value=qrs):
            result = infer_wpw_localization(np.zeros((12, 500), dtype=np.float32), 500.0)
    assert result.site == "coronary_sinus_ostium"


def test_wpw_ptbxl_4825_not_free_wall():
    """Real PTB-XL WPW example should not be labeled right free wall (EP feedback: CS OS)."""
    from synthecg.data.fetch import fetch_ptbxl_record
    from synthecg.data.preprocess import preprocess_record

    record = preprocess_record(fetch_ptbxl_record("records500/04000/04825_hr"), bandpass=True)
    signal = record.p_signal.T
    result = infer_wpw_localization(signal, float(record.fs))
    assert result.site == "coronary_sinus_ostium"
    assert result.site != "right_free_wall"
    assert result.verified is False


def test_infer_localization_afib_not_applicable():
    signal = np.random.randn(12, 500).astype(np.float32) * 0.01
    result = infer_localization_from_signal(signal, 500.0, {"AFIB": 100.0})
    assert result is not None
    assert result.site == "none"
    assert result.level == "none"


def test_load_zheng_database_from_fixture(tmp_path):
    fixture_cache = FIXTURES
    with patch("synthecg.data.zheng_otva.ensure_zheng_data") as mock_ensure:
        mock_ensure.return_value = type(
            "P",
            (),
            {
                "root": fixture_cache,
                "diagnosis_csv": fixture_cache / "diagnosis.csv",
                "ecg_dir": fixture_cache / "ecg_denoised",
            },
        )()
        df = load_zheng_database(fixture_cache, download_ecg=False)
    assert len(df) >= 4
    assert "site" in df.columns
    assert "LCC" in set(df["site"])


def test_select_zheng_records_by_site(tmp_path):
    fixture_cache = FIXTURES
    with patch("synthecg.data.zheng_otva.ensure_zheng_data") as mock_ensure:
        mock_ensure.return_value = type(
            "P",
            (),
            {
                "root": fixture_cache,
                "diagnosis_csv": fixture_cache / "diagnosis.csv",
                "ecg_dir": fixture_cache / "ecg_denoised",
            },
        )()
        df = load_zheng_database(fixture_cache, download_ecg=False)
    lcc = select_zheng_records(df, count=1, site="LCC", seed=0)
    assert len(lcc) == 1
    assert lcc.iloc[0]["site"] == "LCC"


def test_zheng_microvolt_to_millivolt():
    from synthecg.data.zheng_otva import _to_millivolts

    microvolts = np.array([[1500.0, -2000.0]], dtype=np.float32)
    millivolts = _to_millivolts(microvolts)
    assert np.allclose(millivolts, [[1.5, -2.0]])

    already_mv = np.array([[0.5, -1.2]], dtype=np.float32)
    assert np.allclose(_to_millivolts(already_mv), already_mv)


def test_fetch_zheng_record_resamples(tmp_path):
    fixture_cache = FIXTURES
    with patch("synthecg.data.zheng_otva.ensure_zheng_data") as mock_ensure:
        mock_ensure.return_value = type(
            "P",
            (),
            {
                "root": fixture_cache,
                "diagnosis_csv": fixture_cache / "diagnosis.csv",
                "ecg_dir": fixture_cache / "ecg_denoised",
            },
        )()
        df = load_zheng_database(fixture_cache, download_ecg=False)
        hospital_id = int(df[df.site == "LCC"].index[0]) if "LCC" in df.site.values else int(df.index[0])
        record = fetch_zheng_record(hospital_id, cache_dir=fixture_cache)
    assert record.fs == 500.0
    assert record.p_signal.shape[1] == 12


def test_localization_from_row_ep_source():
    row = pd.Series(
        {"Sublocation": "LCC", "LeftRight": "Left", "site": "LCC", "region": "LVOT"},
        name=123,
    )
    loc = localization_from_row(row)
    assert loc.source == "ep_ablation"
    assert loc.site == "LCC"
    assert loc.verified is True


def test_build_annotation_includes_localization():
    from unittest.mock import MagicMock

    record = MagicMock()
    record.fs = 500
    record.sig_len = 5000
    annotation = build_annotation(
        ecg_id=1,
        patient_id=1,
        scp_codes={"PVC": 100.0},
        strat_fold=0,
        record=record,
        render=__import__("synthecg.config", fromlist=["RenderConfig"]).RenderConfig(),
        augment=__import__("synthecg.config", fromlist=["AugmentConfig"]).AugmentConfig(),
        augmentations=[],
        image_path="images/x.png",
        signal_path="signals/x.npy",
        localization={"site": "LCC", "source": "ep_ablation"},
        database="zheng-otva",
    )
    assert annotation["localization"]["site"] == "LCC"
    assert annotation["database"] == "zheng-otva"


def test_arrhythmia_localization_recipe():
    config = config_from_recipe("arrhythmia-localization-v1", output_dir="/tmp/loc", seed=1)
    assert config.database == "zheng-otva"
    assert "LCC" in config.balanced_sites
    assert config.infer_localization is False
