# Examples

This folder contains sample output from SynthECG.

## `sample_12lead_ecg.png`

A synthetic 12-lead ECG image generated from PTB-XL record **#3094** (SCP: NORM, SR).

| Property | Value |
|----------|-------|
| Layout | 3×4 + Lead II rhythm strip |
| Renderer | OpenCV (300 DPI) |
| Paper speed | 25 mm/s |
| Gain | 10 mm/mV |
| Artifacts | None (clean pre-augmentation render) |
| Seed | 42 |

### Reproduce this image

```bash
pip install -e .
synthecg -n 1 -t NORM --seed 42 --split train --save-clean --augment-profile clean -o /tmp/out
# Output: /tmp/out/images/clean/ecg_NORM_3094.png
```
