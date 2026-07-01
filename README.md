# SynthECG-Generator

An open-source Python tool for generating realistic, publication-ready synthetic 12-lead ECG images with **ML-ready ground truth** from the [PTB-XL Database](https://physionet.org/content/ptb-xl/).

Version **0.3.0** adds an OpenCV renderer with calibration pulses and clinical headers, segmentation masks, YOLO bounding boxes, parallel batch generation, and a round-trip digitization benchmark.

## Features

* **Real clinical data:** Uses actual PTB-XL waveforms across all 12 leads.
* **OpenCV renderer (default):** Fixed 300 DPI canvas, medical grid, calibration pulses, headers/footers.
* **Pathology filtering:** Filter by SCP-ECG codes (`AFIB`, `NORM`, `PVC`, `SR`, etc.) or sample randomly.
* **Reproducible datasets:** `--seed`, PTB-XL `--split`, and `--unique-patients`.
* **ML-ready exports:** PNG images, `.npy` signals, JSON annotations, segmentation masks, YOLO labels, and `manifest.csv`.
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

### Parallel batch generation

```bash
synthecg -n 50 -t NORM --seed 42 --workers 4 -o norm_batch
```

### Run digitization benchmark

After generating a dataset with `--save-clean` (recommended for benchmark):

```bash
synthecg-benchmark my_afib_dataset
synthecg-benchmark my_afib_dataset --use-augmented   # test on final augmented images
```

### Arguments

| Flag | Description |
|------|-------------|
| `-n`, `--count` | Number of ECG images to generate (default: `5`) |
| `-t`, `--type` | SCP code such as `NORM`, `AFIB`, `PVC`, or `random` |
| `-o`, `--output-dir` | Output directory (default: `output_ecgs`) |
| `--seed` | Random seed for reproducible sampling and augmentations |
| `--split` | PTB-XL split: `all`, `train` (folds 1–8), `val` (9), `test` (10) |
| `--workers` | Parallel worker processes (default: `1`) |
| `--renderer` | `opencv` (default) or `matplotlib` |
| `--save-clean` | Save pre-augmentation images to `images/clean/` |
| `--augment-profile` | `clean`, `scan`, or `clinical` |
| `--no-masks` | Skip segmentation mask export |
| `--no-yolo` | Skip YOLO label export |

## Output structure

```
my_afib_dataset/
├── manifest.csv
├── benchmark_report.json          # after running synthecg-benchmark
├── images/
│   ├── ecg_AFIB_12345.png
│   └── clean/
│       └── ecg_AFIB_12345.png     # when --save-clean
├── signals/
│   └── ecg_AFIB_12345.npy         # shape (12, n_samples), float32 mV
├── masks/
│   └── ecg_AFIB_12345.png         # waveform segmentation mask
├── labels/
│   ├── classes.txt                # lead_region, lead_label
│   └── ecg_AFIB_12345.txt         # YOLO format bboxes
└── annotations/
    └── ecg_AFIB_12345.json        # metadata, lead bboxes, render params
```

## Annotation JSON

Each annotation includes per-lead bounding boxes for digitization and detection:

```json
{
  "leads": [
    {
      "name": "I",
      "lead_idx": 0,
      "bbox": [80, 120, 738, 575],
      "plot_bbox": [154, 140, 600, 535],
      "baseline_y": 407,
      "t_start": 0.0,
      "t_end": 2.5
    }
  ],
  "render": {
    "backend": "opencv",
    "px_per_mv": 118.11,
    "px_per_second": 295.28
  }
}
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
