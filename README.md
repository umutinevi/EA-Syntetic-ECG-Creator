# SynthECG-Generator

An open-source Python tool for generating realistic, publication-ready synthetic 12-lead ECG images with **ML-ready ground truth** from the [PTB-XL Database](https://physionet.org/content/ptb-xl/).

Version **0.4.0** adds dataset recipes, resume/checkpoint support, advanced filtering, configurable render parameters, and signal bandpass preprocessing.

## Features

* **Real clinical data:** Uses actual PTB-XL waveforms across all 12 leads.
* **OpenCV renderer (default):** Fixed 300 DPI canvas, medical grid, calibration pulses, headers/footers.
* **Configurable rendering:** Paper speed (25/50 mm/s), gain (5/10/20 mm/mV), grid on/off.
* **Signal preprocessing:** Optional 0.5–40 Hz bandpass filter before rendering.
* **Pathology filtering:** Single-code, multi-label include/exclude, and balanced class sampling.
* **Dataset recipes:** Predefined configs for digitization, arrhythmia classification, and clinical scans.
* **Resume support:** `--resume` skips already-generated records in an existing manifest.
* **ML-ready exports:** PNG images, `.npy` signals, JSON annotations, masks, YOLO labels, `manifest.csv`.
* **Parallel generation:** `--workers N` for batch dataset builds.
* **Digitization benchmark:** Round-trip correlation report via `synthecg-benchmark`.

## Installation

```bash
git clone https://github.com/umutinevi/EA-Syntetic-ECG-Creator.git
cd EA-Syntetic-ECG-Creator
pip install -e .
```

## Usage

### Generate a dataset

```bash
synthecg -n 10 -t AFIB --seed 42 --split train --save-clean -o my_afib_dataset
```

Backward-compatible — subcommand is optional:

```bash
synthecg generate -n 10 -t AFIB --seed 42 -o my_afib_dataset
```

### Build from a recipe

```bash
synthecg dataset list
synthecg dataset build --recipe digitization-v1 -o digitization_train --seed 42
synthecg dataset build --recipe arrhythmia-cls -o arrhythmia_train --workers 4
```

| Recipe | Description |
|--------|-------------|
| `digitization-v1` | 100 random train samples, full GT, scan artifacts |
| `arrhythmia-cls` | Balanced AFIB/SR/PVC/NORM (25 per class) |
| `clinical-scan` | 50 samples with perspective warp artifacts |
| `clean-baseline` | 20 clean images for digitization baseline |

### Advanced filtering

```bash
# Require both NORM and SR labels
synthecg -n 20 -t random --include-codes NORM SR -o norm_sr_only

# Exclude atrial fibrillation
synthecg -n 20 -t random --exclude-codes AFIB -o no_afib
```

### Resume interrupted generation

```bash
synthecg -n 100 -t NORM --seed 42 -o big_dataset --resume
```

### Configurable rendering

```bash
synthecg -n 5 -t NORM --speed 50 --gain 5 --no-grid --bandpass -o fast_gain5
```

### Run digitization benchmark

```bash
synthecg-benchmark my_afib_dataset
synthecg-benchmark my_afib_dataset --use-augmented
```

## Output structure

```
my_afib_dataset/
├── manifest.csv
├── benchmark_report.json
├── images/ (+ images/clean/ with --save-clean)
├── signals/
├── masks/
├── labels/
└── annotations/
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Scientific use

This pipeline helps the ML and computer vision community build realistic ECG image datasets for digitization and classification without private clinical data constraints.

**Data source:** Wagner, P., et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*.

## License

MIT — see [LICENSE](LICENSE).
