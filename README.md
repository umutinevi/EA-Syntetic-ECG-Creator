# SynthECG-Generator

An open-source Python tool for generating realistic, publication-ready synthetic 12-lead ECG images with **ML-ready ground truth** from the [PTB-XL Database](https://physionet.org/content/ptb-xl/).

This tool fetches real clinical ECG signals from PhysioNet, plots them onto a standard medical grid (25 mm/s, 10 mm/mV), applies scan-style augmentations, and exports paired images, signals, annotations, and a dataset manifest.

## Features

* **Real clinical data:** Uses actual PTB-XL waveforms across all 12 leads.
* **Pathology filtering:** Filter by SCP-ECG codes (`AFIB`, `NORM`, `PVC`, `SR`, etc.) or sample randomly.
* **Reproducible datasets:** `--seed`, PTB-XL `--split`, and `--unique-patients` support.
* **ML-ready exports:** PNG images, `.npy` signals, JSON annotations, and `manifest.csv`.
* **Cached metadata:** PTB-XL CSV is cached locally after the first download.
* **Classic medical layout:** 3×4 + Lead II rhythm strip.
* **Configurable artifacts:** `clean`, `scan` (default), or `clinical` (scan + perspective warp).

## Installation

```bash
git clone https://github.com/umutinevi/EA-Syntetic-ECG-Creator.git
cd EA-Syntetic-ECG-Creator
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

## Usage

### Recommended (package CLI)

```bash
synthecg -n 10 -t AFIB --seed 42 --split train -o my_afib_dataset
```

Equivalent module invocation:

```bash
python -m synthecg -n 10 -t AFIB --seed 42 --split train -o my_afib_dataset
```

### Backward-compatible script

```bash
python generate_realistic_ecg.py -n 10 -t AFIB -o my_afib_dataset
```

### Arguments

| Flag | Description |
|------|-------------|
| `-n`, `--count` | Number of ECG images to generate (default: `5`) |
| `-t`, `--type` | SCP code such as `NORM`, `AFIB`, `PVC`, or `random` (default) |
| `-o`, `--output-dir` | Output directory (default: `output_ecgs`) |
| `--seed` | Random seed for reproducible sampling and augmentations |
| `--split` | PTB-XL split: `all`, `train` (folds 1–8), `val` (9), `test` (10) |
| `--cache-dir` | Custom cache directory for PTB-XL metadata CSV |
| `--unique-patients` | Sample at most one record per patient |
| `--augment-profile` | `clean`, `scan`, or `clinical` |
| `--no-signals` | Skip `.npy` signal export |
| `--no-annotations` | Skip JSON annotation export |

## Output structure

```
my_afib_dataset/
├── manifest.csv
├── images/
│   └── ecg_AFIB_12345.png
├── signals/
│   └── ecg_AFIB_12345.npy      # shape (12, n_samples), float32 mV
└── annotations/
    └── ecg_AFIB_12345.json     # metadata, render params, paths
```

### Manifest columns

`sample_id`, `ecg_id`, `patient_id`, `diagnosis_query`, `scp_codes`, `strat_fold`, `image_path`, `signal_path`, `annotation_path`, `augmentations`

## Example

```bash
synthecg -n 3 -t NORM --seed 42 --split train -o demo_dataset
```

First run downloads and caches PTB-XL metadata (~30s). Subsequent runs use the local cache.

## Scientific use

This pipeline helps the ML and computer vision community build realistic ECG image datasets for digitization and classification without private clinical data constraints.

**Data source:** Wagner, P., et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
