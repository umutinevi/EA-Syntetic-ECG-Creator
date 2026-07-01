"""Automated Hugging Face publish pipeline for SynthECG."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from synthecg.benchmark.digitize import run_benchmark
from synthecg.export.huggingface import HF_TOKEN_ENV, prepare_hf_export, push_as_hf_dataset, push_folder_to_hub
from synthecg.pipeline import generate_dataset
from synthecg.recipes.builder import config_from_recipe


@dataclass
class HFPublishConfig:
    """Configuration for automated generate → export → publish workflow."""

    repo_id: str
    output_dir: str = "hf_dataset"
    recipe: str | None = None
    count: int | None = None
    diagnosis: str = "random"
    split: Literal["all", "train", "val", "test"] = "train"
    seed: int = 42
    workers: int | None = None
    resume: bool = False
    layout: str | None = None
    push: bool = True
    private: bool = False
    benchmark: bool = False
    benchmark_use_clean: bool = True
    format: Literal["folder", "datasets"] = "folder"
    export_dir: str | None = None
    token: str | None = None
    generation_overrides: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> HFPublishConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML config: {path}")
        return cls(**data)

    def to_dict(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "output_dir": self.output_dir,
            "recipe": self.recipe,
            "count": self.count,
            "diagnosis": self.diagnosis,
            "split": self.split,
            "seed": self.seed,
            "workers": self.workers,
            "resume": self.resume,
            "layout": self.layout,
            "push": self.push,
            "private": self.private,
            "benchmark": self.benchmark,
            "benchmark_use_clean": self.benchmark_use_clean,
            "format": self.format,
            "export_dir": self.export_dir,
        }


def _generate_if_needed(config: HFPublishConfig) -> Path:
    output_dir = Path(config.output_dir)

    if config.recipe:
        gen_config = config_from_recipe(
            config.recipe,
            output_dir=str(output_dir),
            seed=config.seed,
            resume=config.resume,
            workers=config.workers,
        )
        if config.count is not None:
            gen_config.count = config.count
        for key, value in config.generation_overrides.items():
            if hasattr(gen_config, key):
                setattr(gen_config, key, value)
        print(f"[1/4] Generating dataset from recipe '{config.recipe}' -> {output_dir}")
        generate_dataset(gen_config)
        return output_dir

    if not (output_dir / "manifest.csv").exists():
        raise FileNotFoundError(
            f"No manifest.csv in {output_dir}. Provide --recipe or generate a dataset first."
        )

    print(f"[1/4] Using existing dataset at {output_dir}")
    return output_dir


def publish_to_hub(config: HFPublishConfig) -> dict:
    """Run the full automated pipeline: generate → benchmark → export → push."""
    token = config.token or os.environ.get(HF_TOKEN_ENV)
    if config.push and not token:
        raise EnvironmentError(
            f"HF token required for --push. Set the {HF_TOKEN_ENV} environment variable."
        )

    dataset_dir = _generate_if_needed(config)

    benchmark_summary = None
    if config.benchmark:
        print("[2/4] Running digitization benchmark ...")
        benchmark_summary = run_benchmark(
            dataset_dir,
            use_clean=config.benchmark_use_clean,
        )
        mean_corr = benchmark_summary.get("mean_correlation")
        print(f"      Mean correlation: {mean_corr:.4f}" if mean_corr == mean_corr else "      Benchmark complete.")
    else:
        print("[2/4] Skipping benchmark (disabled)")

    export_path = config.export_dir or str(Path(dataset_dir) / "hf_export")
    print(f"[3/4] Preparing Hugging Face export -> {export_path}")
    prepared = prepare_hf_export(
        dataset_dir,
        export_dir=export_path,
        layout=config.layout,
        repo_id=config.repo_id,
    )

    result = {
        "dataset_dir": str(dataset_dir),
        "export_dir": str(prepared),
        "repo_id": config.repo_id,
        "hub_url": None,
        "benchmark": benchmark_summary,
        "pushed": False,
    }

    if config.push:
        print(f"[4/4] Pushing to Hugging Face Hub: {config.repo_id} (format={config.format})")
        if config.format == "datasets":
            url = push_as_hf_dataset(
                dataset_dir,
                config.repo_id,
                token=token,
                private=config.private,
            )
        else:
            url = push_folder_to_hub(
                prepared,
                config.repo_id,
                token=token,
                private=config.private,
            )
        result["hub_url"] = url
        result["pushed"] = True
        print(f"      Published: {url}")
    else:
        print("[4/4] Skipping push (--no-push). Export ready locally.")

    report_path = Path(dataset_dir) / "hf_publish_report.json"
    report_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Publish report: {report_path}")
    return result
