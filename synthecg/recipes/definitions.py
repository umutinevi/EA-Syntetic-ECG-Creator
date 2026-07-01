"""Predefined dataset generation recipes."""

from __future__ import annotations

from typing import Any

RECIPES: dict[str, dict[str, Any]] = {
    "digitization-v1": {
        "description": "100 random train samples with full ground truth for digitization research.",
        "count": 100,
        "split": "train",
        "diagnosis": "random",
        "augment_profile": "scan",
        "save_clean": True,
        "workers": 2,
        "bandpass_filter": True,
        "render": {
            "speed_mm_s": 25,
            "gain_mm_mv": 10,
            "show_grid": True,
            "backend": "opencv",
        },
    },
    "arrhythmia-cls": {
        "description": "Balanced AFIB/SR/PVC/NORM set for arrhythmia classification (25 per class).",
        "balanced_codes": ["AFIB", "SR", "PVC", "NORM"],
        "count_per_code": 25,
        "split": "train",
        "augment_profile": "scan",
        "save_clean": True,
        "workers": 2,
        "bandpass_filter": True,
    },
    "clinical-scan": {
        "description": "50 samples with clinical scan artifacts including perspective warps.",
        "count": 50,
        "split": "train",
        "diagnosis": "random",
        "augment_profile": "clinical",
        "save_clean": True,
        "workers": 2,
        "bandpass_filter": True,
    },
    "clean-baseline": {
        "description": "20 clean (no artifact) samples for digitization baseline benchmarking.",
        "count": 20,
        "split": "train",
        "diagnosis": "random",
        "augment_profile": "clean",
        "save_clean": True,
        "workers": 1,
        "bandpass_filter": True,
        "render": {
            "speed_mm_s": 25,
            "gain_mm_mv": 10,
            "show_grid": True,
        },
    },
    "digitization-12x1": {
        "description": "50 train samples in 12x1 stacked layout for vertical lead digitization.",
        "count": 50,
        "split": "train",
        "diagnosis": "random",
        "augment_profile": "scan",
        "save_clean": True,
        "workers": 2,
        "bandpass_filter": True,
        "render": {
            "layout": "12x1",
            "speed_mm_s": 25,
            "gain_mm_mv": 10,
            "show_grid": True,
            "backend": "opencv",
        },
    },
}


def list_recipes() -> dict[str, str]:
    return {name: recipe["description"] for name, recipe in RECIPES.items()}


def get_recipe(name: str) -> dict[str, Any]:
    if name not in RECIPES:
        available = ", ".join(sorted(RECIPES))
        raise ValueError(f"Unknown recipe '{name}'. Available: {available}")
    return RECIPES[name].copy()
