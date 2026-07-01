# Arrhythmia Localization in SynthECG

SynthECG v0.7 adds **provenance-aware arrhythmia localization** — anatomical or mechanism labels tied to either EP-validated datasets or literature-based ECG algorithms.

## The PTB-XL gap

PTB-XL provides SCP-ECG diagnostic codes (`AFIB`, `PVC`, `WPW`, `PSVT`) but **no electrophysiology mapping labels**. Coronary cusp, RVOT free wall, and accessory pathway locations cannot be read from PTB-XL metadata alone.

## Hybrid strategy

| Arrhythmia | Primary source | Fallback |
|---|---|---|
| PVC (OT-VA) | **Zheng OT-VA dataset** (EP ablation ground truth) | `pvc_otva_v1` algorithm on PTB-XL |
| WPW | `wpw_arruda_simplified_v1` algorithm | Manual curation |
| AVNRT | PTB-XL `PSVT` proxy | Mechanism label only (`slow_fast_avnrt`) |
| AFIB | PTB-XL | No localization (`none`) |

Every localization block in annotations includes:

- `source`: `ep_ablation` | `algorithm` | `manual_curated`
- `confidence`: 0–1 float
- `region` / `site` / `site_label`
- `taxonomy_version`: `synthecg-v1`

## Zheng OT-VA dataset

**Reference:** Zheng et al., *Scientific Data* 2020 — [doi:10.1038/s41597-020-0440-8](https://doi.org/10.1038/s41597-020-0440-8)

334 patients with idiopathic outflow tract PVC/VT and **catheter ablation–validated origins**:

| Region | Sublocations |
|---|---|
| LVOT (6) | LCC, RCC, NCC, AMC, Summit, LCC-RCC commissure |
| RVOT (7) | LC, RC, AC, free wall, anterior/posterior septal, other |

### Download

The adapter caches data under `~/.cache/synthecg/zheng_otva/` on first use:

```bash
# Generate from Zheng EP-validated records (downloads ~694 MB ECG zip once)
synthecg generate -n 5 -t LCC --database zheng-otva -o zheng_lcc

# Balanced localization recipe
synthecg dataset build --recipe arrhythmia-localization-v1 -o otva_localization
```

## Literature algorithms

### PVC / OT-VA (`pvc_otva_v1`)

1. Isolate ectopic vs sinus beat (R-peak energy)
2. **V2 transition ratio** (Betensky) → RVOT vs LVOT (threshold 0.6)
3. **V1 r-wave** hint → LCC when LVOT

### WPW (`wpw_arruda_simplified_v1`)

Simplified Arruda-style delta polarity in leads I, II, aVF, V1 plus R/S ratios → accessory pathway region (e.g. `right_free_wall`).

### Benchmark

Validate algorithms against Zheng EP labels:

```bash
synthecg-localize --limit 50 -o localization_benchmark.json
```

## Annotation schema

```json
{
  "localization": {
    "level": "sublocation",
    "region": "LVOT",
    "site": "LCC",
    "site_label": "Left Coronary Cusp",
    "source": "ep_ablation",
    "confidence": 1.0,
    "taxonomy_version": "synthecg-v1"
  }
}
```

Algorithm-inferred labels additionally include `algorithm`, `features`, and `decision_path`.

## Examples gallery

Regenerate provenance-aware examples:

```bash
python scripts/generate_arrhythmia_examples.py
```

- **PVC LCC** — real Zheng EP-validated record
- **WPW right free wall** — best PTB-XL match via algorithm
- **AVNRT** — PSVT proxy with mechanism metadata
- **AFIB** — rhythm only, no fake anatomy

See `examples/arrhythmia_index.json` for full metadata per image.

## Limitations

- WPW localization on PTB-XL is **algorithm-inferred**, not ablation-confirmed.
- AVNRT has no public 12-lead + EP mapping corpus; only mechanism-level proxy labels.
- AFIB has no meaningful single anatomic origin on surface ECG.
- Always check `localization.source` before using labels as ground truth for training.
