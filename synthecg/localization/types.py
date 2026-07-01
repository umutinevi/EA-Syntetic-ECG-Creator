"""Localization metadata types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from synthecg.localization.taxonomy import TAXONOMY_VERSION, site_label


@dataclass
class LocalizationInfo:
    level: str
    region: str | None
    site: str | None
    site_label: str
    source: str
    confidence: float
    taxonomy_version: str = TAXONOMY_VERSION
    algorithm: str | None = None
    features: dict = field(default_factory=dict)
    decision_path: list[str] = field(default_factory=list)

    @classmethod
    def from_ep(
        cls,
        *,
        region: str,
        site: str,
        confidence: float = 1.0,
    ) -> LocalizationInfo:
        return cls(
            level="sublocation",
            region=region,
            site=site,
            site_label=site_label(site),
            source="ep_ablation",
            confidence=confidence,
        )

    @classmethod
    def from_algorithm(
        cls,
        *,
        region: str | None,
        site: str | None,
        algorithm: str,
        confidence: float,
        features: dict | None = None,
        decision_path: list[str] | None = None,
        level: str = "sublocation",
    ) -> LocalizationInfo:
        label = site_label(site) if site else "Unknown"
        return cls(
            level=level,
            region=region,
            site=site,
            site_label=label,
            source="algorithm",
            confidence=confidence,
            algorithm=algorithm,
            features=features or {},
            decision_path=decision_path or [],
        )

    @classmethod
    def not_applicable(cls, reason: str = "Rhythm has no discrete anatomic origin") -> LocalizationInfo:
        return cls(
            level="none",
            region=None,
            site="none",
            site_label=site_label("none"),
            source="manual_curated",
            confidence=1.0,
            features={"note": reason},
        )

    def to_dict(self) -> dict:
        return asdict(self)
