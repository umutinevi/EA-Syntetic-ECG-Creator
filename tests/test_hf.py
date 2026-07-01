import json
from pathlib import Path
from unittest.mock import patch

import pytest

from synthecg.export.huggingface import (
    build_dataset_card,
    build_metadata_rows,
    detect_layout,
    prepare_hf_export,
    write_metadata_parquet,
)
from synthecg.export.publish import HFPublishConfig, publish_to_hub

pytest.importorskip("pyarrow")


MANIFEST_HEADER = (
    "sample_id,ecg_id,patient_id,diagnosis_query,scp_codes,strat_fold,"
    "image_path,signal_path,annotation_path,mask_path,yolo_path,clean_image_path,augmentations\n"
)


def _setup_dataset(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "images").mkdir()
    (tmp_path / "signals").mkdir()
    (tmp_path / "annotations").mkdir()

    sample_id = "ecg_NORM_1"
    (tmp_path / "images" / f"{sample_id}.png").write_bytes(b"png")
    import numpy as np

    np.save(tmp_path / "signals" / f"{sample_id}.npy", np.zeros((12, 100), dtype=np.float32))
    annotation = {
        "render": {"layout": "3x4+1"},
        "augment": {"profile": "clean"},
    }
    (tmp_path / "annotations" / f"{sample_id}.json").write_text(json.dumps(annotation), encoding="utf-8")

    (tmp_path / "manifest.csv").write_text(
        MANIFEST_HEADER
        + f'{sample_id},1,10,NORM,"{{""NORM"": 80.0}}",1,images/{sample_id}.png,'
        f"signals/{sample_id}.npy,annotations/{sample_id}.json,,,,\n",
        encoding="utf-8",
    )
    return tmp_path


def test_build_metadata_rows(tmp_path):
    dataset = _setup_dataset(tmp_path)
    rows = build_metadata_rows(dataset)
    assert len(rows) == 1
    assert rows[0]["ecg_id"] == 1
    assert rows[0]["layout"] == "3x4+1"
    assert rows[0]["signal_shape"] == [12, 100]


def test_detect_layout(tmp_path):
    dataset = _setup_dataset(tmp_path)
    assert detect_layout(dataset) == "3x4+1"


def test_write_metadata_parquet(tmp_path):
    dataset = _setup_dataset(tmp_path)
    out = write_metadata_parquet(dataset, tmp_path / "export")
    assert out.exists()


def test_prepare_hf_export_includes_parquet(tmp_path):
    dataset = _setup_dataset(tmp_path)
    out = prepare_hf_export(dataset, export_dir=tmp_path / "hf", repo_id="user/demo")
    assert (out / "metadata.parquet").exists()
    assert (out / "README.md").exists()
    card = (out / "README.md").read_text(encoding="utf-8")
    assert "user/demo" in card


def test_publish_no_push(tmp_path):
    dataset = _setup_dataset(tmp_path)
    config = HFPublishConfig(
        repo_id="user/test",
        output_dir=str(dataset),
        push=False,
        benchmark=False,
    )
    result = publish_to_hub(config)
    assert result["pushed"] is False
    assert Path(result["export_dir"]).exists()
    assert (dataset / "hf_publish_report.json").exists()


def test_publish_from_yaml(tmp_path):
    yaml = pytest.importorskip("yaml")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "repo_id": "user/yaml-test",
                "output_dir": str(_setup_dataset(tmp_path / "data")),
                "push": False,
                "benchmark": False,
            }
        ),
        encoding="utf-8",
    )
    config = HFPublishConfig.from_yaml(config_path)
    assert config.repo_id == "user/yaml-test"
