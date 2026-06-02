# SynthECG-Generator

An open-source Python tool for generating highly realistic, publication-ready synthetic 12-lead ECG images. 

This tool fetches real clinical ECG signals from the massive [PTB-XL Database](https://physionet.org/content/ptb-xl/) via PhysioNet, plots them onto a mathematically accurate standard medical grid (25 mm/s, 10 mm/mV), and applies computer vision augmentations to simulate the classic warm-pink paper aesthetic of scanned/printed ECGs.

## Features
* **Real Clinical Data:** Uses actual patient data ensuring perfect physiological realism across all 12 leads.
* **Pathology Filtering:** Generate specific arrhythmias or pathologies (e.g., `AFIB`, `NORM`, `PVC`, `STEMI`) by leveraging the PTB-XL diagnostic labels.
* **Classic Medical Grid:** Accurately renders a true A4/Letter landscape 3x4 + 1 Rhythm layout.
* **Realistic Artifacts:** Simulates paper grain, uneven lighting, and slight scanner perspective warps.

## Installation

1. Clone this repository.
2. Create a virtual environment and install the dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Use the command-line interface to generate ECGs. The script will automatically fetch the necessary metadata and signals from PhysioNet.

```bash
python generate_realistic_ecg.py --count 10 --type AFIB --output-dir my_afib_dataset
```

### Arguments
* `-n`, `--count`: Number of ECG images to generate (default: `5`).
* `-t`, `--type`: The type of pathology to generate. Use `random` for any, or provide a PTB-XL SCP-ECG code such as `NORM` (Normal), `AFIB` (Atrial Fibrillation), `PVC` (Premature Ventricular Contraction), `SR` (Sinus Rhythm). (default: `random`).
* `-o`, `--output-dir`: The directory to save the generated images (default: `output_ecgs`).

## Example Outputs
The generator uses a specifically tuned OpenCV pipeline to match the classic aesthetic found in clinical reference libraries. 

* *Note: First runs will take a few seconds to download the database index CSV from PhysioNet.*

## Scientific Use
This pipeline was built to aid the machine learning and computer vision community in creating robust, realistic datasets for ECG digitization and classification tasks without running into data privacy and sharing constraints.

*Data Source: Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F.I., Samek, W., Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset.*
