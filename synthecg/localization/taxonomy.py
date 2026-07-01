"""Shared arrhythmia localization taxonomy for SynthECG."""

from __future__ import annotations

import math

TAXONOMY_VERSION = "synthecg-v1"

# Zheng OT-VA dataset sublocation labels -> canonical site ids.
ZHENG_SITE_MAP: dict[str, str] = {
    "LCC": "LCC",
    "RCC": "RCC",
    "NCC": "NCC",
    "AMC": "AMC",
    "Summit": "Summit",
    "LCC-RCC Ommisure": "LCC-RCC_commissure",
    "LC": "LC",
    "RC": "RC",
    "AC": "AC",
    "FreeWall": "free_wall",
    "AnteriorSeptal": "anterior_septal",
    "PosteriorSeptal": "posterior_septal",
    "RVOTOther": "other",
}

SITE_LABELS: dict[str, str] = {
    "LCC": "Left Coronary Cusp",
    "RCC": "Right Coronary Cusp",
    "NCC": "Non-Coronary Cusp",
    "AMC": "Aortomitral Continuity",
    "Summit": "Left Ventricular Summit",
    "LCC-RCC_commissure": "LCC-RCC Commissure",
    "LC": "Left Cusp (RVOT)",
    "RC": "Right Cusp (RVOT)",
    "AC": "Anterior Cusp (RVOT)",
    "free_wall": "Right Ventricular Free Wall",
    "anterior_septal": "Anterior Septal (RVOT)",
    "posterior_septal": "Posterior Septal (RVOT)",
    "other": "Other RVOT",
    "right_lateral": "Right Lateral Accessory Pathway",
    "right_free_wall": "Right Free Wall Accessory Pathway",
    "left_lateral": "Left Lateral Accessory Pathway",
    "left_septal": "Left Septal Accessory Pathway",
    "posteroseptal": "Posteroseptal Accessory Pathway",
    "anteroseptal": "Anteroseptal Accessory Pathway",
    "slow_fast_avnrt": "Slow-Fast AVNRT",
    "unknown_mechanism": "Unknown SVT Mechanism",
    "none": "Not Applicable",
}

LVOT_SITES = {"LCC", "RCC", "NCC", "AMC", "Summit", "LCC-RCC_commissure"}
RVOT_SITES = {"LC", "RC", "AC", "free_wall", "anterior_septal", "posterior_septal", "other"}
WPW_SITES = {
    "right_lateral",
    "right_free_wall",
    "left_lateral",
    "left_septal",
    "posteroseptal",
    "anteroseptal",
}


def normalize_zheng_site(raw: str) -> str:
    """Map a Zheng diagnostics Sublocation string to a canonical site id."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        raise ValueError("Missing Zheng sublocation")
    key = str(raw).strip()
    if key.upper() in {"NA", "NAN", ""}:
        raise ValueError(f"Missing Zheng sublocation: {raw!r}")
    if key not in ZHENG_SITE_MAP:
        raise ValueError(f"Unknown Zheng sublocation: {raw!r}")
    return ZHENG_SITE_MAP[key]


def region_for_site(site: str) -> str | None:
    if site in LVOT_SITES:
        return "LVOT"
    if site in RVOT_SITES:
        return "RVOT"
    if site in WPW_SITES:
        return "AP"
    return None


def site_label(site: str) -> str:
    return SITE_LABELS.get(site, site.replace("_", " ").title())
