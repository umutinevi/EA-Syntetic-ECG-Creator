# SynthECG-Generator

An open-source Python tool for generating realistic, publication-ready synthetic 12-lead ECG images with **ML-ready ground truth** from the [PTB-XL Database](https://physionet.org/content/ptb-xl/).

Version **0.5.0** adds a 12×1 layout, enhanced scan artifacts, Hugging Face export, and CI.

## Features

* **Layouts:** `3x4+1` (classic) and `12x1` (vertical stack, full 10 s per lead)
* **OpenCV renderer:** 300 DPI canvas, medical grid, calibration pulses, headers/footers
* **Enhanced artifacts:** rotation, JPEG compression, perspective warp (clinical profile)
* **Dataset recipes:** digitization, arrhythmia classification, 12×1 layout
* **Hugging Face export:** prepare or push datasets to the Hub
* **CI:** GitHub Actions test suite on Python 3.10–3.12
* **ML-ready exports:** images, signals, annotations, masks, YOLO labels, manifest

## Installation

```bash
git clone https://github.com/umutinevi/EA-Syntetic-ECG-Creator.git
cd EA-Syntetic-ECG-Creator
pip install -e .
```

Optional Hugging Face upload support:

```bash
pip install -e ".[hf]"
```

## Usage

### Generate with layout

```bash
synthecg -n 10 -t NORM --layout 12x1 --save-clean -o stack_dataset
synthecg -n 10 -t AFIB --layout 3x4+1 -o classic_dataset
```

### Build from recipe

```bash
synthecg dataset list
synthecg dataset build --recipe digitization-12x1 -o stack_train --seed 42
```

### Export to Hugging Face

```bash
# Prepare local HF folder (README + dataset_info.json)
synthecg export hf -d my_dataset -o my_dataset/hf_export --layout 3x4+1

# Upload (requires HF token in HF_TOKEN env var)
pip install -e ".[hf]"
synthecg export hf -d my_dataset --repo-id your-user/synthecg-demo --push
```

### Benchmark

```bash
synthecg-benchmark my_dataset
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
