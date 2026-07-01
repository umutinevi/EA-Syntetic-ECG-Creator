# Arrhythmia Localization in SynthECG

SynthECG v0.7 adds **provenance-aware arrhythmia localization** — anatomical or mechanism labels tied to either EP-validated datasets or literature-based ECG algorithms.

## The PTB-XL gap

PTB-XL provides SCP-ECG diagnostic codes (`AFIB`, `PVC`, `WPW`, `PSVT`) but **no electrophysiology mapping labels**. Coronary cusp, RVOT free wall, and accessory pathway locations cannot be read from PTB-XL metadata alone.

## Hybrid strategy

| Arrhythmia | Primary source | Fallback |
|---|---|---|
| PVC (OT-VA) | **Zheng OT-VA dataset** (EP ablation ground truth) | `pvc_otva_literature_v2` on PTB-XL |
| WPW | `wpw_arruda_milstein_v2` literature rules | **Unverified** — requires EP confirmation |
| AVNRT | PTB-XL `PSVT` proxy | Mechanism label only (`slow_fast_avnrt`) |
| AFIB | PTB-XL | No localization (`none`) |

Every localization block includes:

- `source`: `ep_ablation` | `algorithm` | `manual_curated`
- `verified`: `true` only for EP ablation ground truth
- `confidence`: capped at **0.75** (PVC algorithm) and **0.50** (WPW algorithm) until clinically validated
- `literature_references`: DOIs / citations for the rules applied
- `region` / `site` / `site_label`

## Zheng OT-VA dataset

**Reference:** Zheng et al., *Scientific Data* 2020 — [doi:10.1038/s41597-020-0440-8](https://doi.org/10.1038/s41597-020-0440-8)

334 patients with idiopathic outflow tract PVC/VT and **catheter ablation–validated origins**.

## Literature algorithms

### PVC / OT-VA (`pvc_otva_literature_v2`)

Multi-step pipeline from published criteria:

1. **Betensky V2 transition ratio** (≥ 0.6 → LVOT) — JACC 2011
2. **Yoshida V2S/V3R index** (< 1.5 supports LVOT) — JCE 2014
3. **LCC morphology** — small r wave in V1 during ectopic beat
4. **Di V1–V3 transition index** (> −1.60 → RVOT septal vs lateral) — JCE 2021

Confidence capped at **0.75** (`verified=false`).

### WPW (`wpw_arruda_milstein_v2`)

Combined rules from:

- **Arruda 1998** — negative delta in lead II → coronary venous / **CS OS** region
- **Milstein / Chiang 2008** — inferior lead delta polarity + V1 R/S for septal vs lateral

Important: **I− / II+ / V1− with low R/S** maps to **coronary sinus ostium**, not right free wall.

Confidence capped at **0.50** (`verified=false`). Not for clinical use without EP confirmation.

### Benchmark (PVC only today)

```bash
synthecg-localize --limit 50 -o localization_benchmark.json
```

WPW requires a future EP-labeled cohort — PTB-XL has no ablation sites.

## Annotation schema

```json
{
  "localization": {
    "level": "sublocation",
    "region": "AP",
    "site": "coronary_sinus_ostium",
    "site_label": "Coronary Sinus Ostium (CS OS)",
    "source": "algorithm",
    "verified": false,
    "confidence": 0.47,
    "algorithm": "wpw_arruda_milstein_v2",
    "literature_references": ["Arruda MS et al. JCE 1998 ..."],
    "taxonomy_version": "synthecg-v1"
  }
}
```

## Examples gallery

```bash
python scripts/generate_arrhythmia_examples.py
```

- **PVC LCC** — Zheng EP-validated (`verified=true`)
- **WPW** — `wpw_accessory_pathway.png`, literature prediction only (`verified=false`)
- **AVNRT** — PSVT proxy with mechanism metadata
- **AFIB** — rhythm only

## Limitations

- WPW algorithm labels are **hypotheses** — EP review may disagree (CS OS vs free wall).
- Never train on WPW sublocation labels from PTB-XL without EP ground truth.
- Always check `localization.verified` before using labels as ML ground truth.
