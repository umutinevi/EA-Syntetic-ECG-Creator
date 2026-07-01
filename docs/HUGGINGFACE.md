# Hugging Face Automation Guide

This guide covers how to automatically generate, export, and publish SynthECG datasets to the [Hugging Face Hub](https://huggingface.co/datasets).

---

## Prerequisites

```bash
pip install -e ".[hf]"
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **write** permission.

---

## Quick publish

```bash
python -m synthecg.hf publish \
  --recipe clean-baseline \
  --repo-id YOUR_USERNAME/synthecg-demo \
  --output-dir ./datasets/demo \
  --benchmark \
  --push
```

This runs four steps automatically:

1. **Generate** 20 samples using the `clean-baseline` recipe  
2. **Benchmark** round-trip digitization quality  
3. **Export** HF-ready folder with `metadata.parquet` and dataset card  
4. **Push** to `https://huggingface.co/datasets/YOUR_USERNAME/synthecg-demo`

---

## YAML-driven automation

For repeatable pipelines, use a config file:

```bash
cp configs/hf_publish.example.yaml configs/my_publish.yaml
# Edit repo_id, recipe, output_dir
python -m synthecg.hf publish --config configs/my_publish.yaml
```

### Example config

```yaml
repo_id: your-username/synthecg-digitization-v1
output_dir: ./datasets/digitization-v1
recipe: digitization-v1
seed: 42
workers: 4
split: train
push: true
private: false
benchmark: true
benchmark_use_clean: true
format: folder
```

---

## CLI reference

```bash
python -m synthecg.hf publish [options]
synthecg-hf publish [options]
```

| Flag | Description |
|------|-------------|
| `--config` | YAML config file (overrides other flags) |
| `--recipe` | Generate from a named recipe before export |
| `-d`, `--dataset-dir` | Use existing dataset (skip generation) |
| `-o`, `--output-dir` | Output directory for generation |
| `--repo-id` | Hugging Face dataset id (`user/name`) |
| `-n`, `--count` | Override recipe sample count |
| `--seed` | Random seed (default: 42) |
| `--workers` | Parallel generation workers |
| `--split` | PTB-XL split: train / val / test |
| `--layout` | Force layout in dataset card |
| `--resume` | Skip already-generated samples |
| `--benchmark` | Run digitization benchmark before export |
| `--private` | Create a private HF repo |
| `--format` | `folder` (full assets) or `datasets` (HF API) |
| `--push` | Upload to Hugging Face Hub |
| `--no-push` | Prepare export locally only |
| `--export-dir` | Custom local HF export path |

---

## GitHub Actions

The repo includes `.github/workflows/publish-hf.yml` for manual publishing from CI.

### Setup

1. Open your GitHub repo → **Settings → Secrets and variables → Actions**
2. Add secret: `HF_TOKEN` = your Hugging Face write token
3. Go to **Actions → Publish to Hugging Face → Run workflow**

### Inputs

| Input | Description |
|-------|-------------|
| `recipe` | SynthECG recipe name |
| `repo_id` | Target HF dataset repo |
| `count` | Optional sample count override |
| `push` | Whether to upload to Hub |

The workflow uploads `hf_dataset/` as a downloadable artifact even if push fails.

---

## Upload formats explained

### `folder` (default)

Uploads the complete `hf_export/` directory:

- All PNG images, `.npy` signals, JSON annotations, masks, YOLO labels
- `metadata.parquet` for tabular indexing
- Auto-generated `README.md` dataset card

Best for: full digitization research, users who need signals and masks.

### `datasets`

Uses the Hugging Face `datasets` library to push a structured dataset with an `Image` column and metadata fields.

Best for: quick exploration in HF UI, image classification demos.

```bash
python -m synthecg.hf publish \
  --recipe clean-baseline \
  --repo-id YOUR_USERNAME/synthecg-images \
  --format datasets \
  --push
```

---

## Publish an existing local dataset

If you already generated a dataset:

```bash
python -m synthecg.hf publish \
  -d ./my_dataset \
  --repo-id YOUR_USERNAME/my-ecg-dataset \
  --benchmark \
  --push
```

---

## Outputs after publish

```
my_dataset/
├── hf_export/                  # HF-ready export folder
│   ├── README.md
│   ├── metadata.parquet
│   └── images/ signals/ ...
└── hf_publish_report.json      # Automation report
```

Example report:

```json
{
  "dataset_dir": "./datasets/demo",
  "export_dir": "./datasets/demo/hf_export",
  "repo_id": "your-username/synthecg-demo",
  "hub_url": "https://huggingface.co/datasets/your-username/synthecg-demo",
  "pushed": true
}
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `HF token required` | `export HF_TOKEN=hf_...` |
| `Install HF support` | `pip install -e ".[hf]"` |
| Repo already exists | Safe — upload uses `exist_ok=True` |
| Large upload timeout | Use `--format datasets` or reduce `--count` |
| Private dataset | Add `--private` |

---

## Recommended recipes for HF publishing

| Recipe | Samples | Best for |
|--------|---------|----------|
| `clean-baseline` | 20 | Quick demo / sanity check |
| `digitization-v1` | 100 | Digitization research |
| `arrhythmia-cls` | 100 | Classification (balanced) |
| `digitization-12x1` | 50 | Vertical layout digitization |

```bash
python -m synthecg.hf publish \
  --recipe digitization-v1 \
  --repo-id YOUR_USERNAME/synthecg-digitization-v1 \
  --workers 4 \
  --benchmark \
  --push
```
