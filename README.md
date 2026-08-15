# SciFigAlign

**Scoring Scientific Figures by Fine-tuned Alignment of Visuals with Manuscript Evidence**

[![arXiv](https://img.shields.io/badge/arXiv-2607.27066-b31b1b.svg)](https://arxiv.org/abs/2607.27066)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-FrankDengAI%2FSciFigAlign-181717?logo=github)](https://github.com/FrankDengAI/SciFigAlign)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-haihanlamu%2FSciFigAlign-ffcc4d)](https://huggingface.co/datasets/haihanlamu/SciFigAlign)
[![Test MAE](https://img.shields.io/badge/test%20MAE-0.3524-0e7c66)](https://arxiv.org/abs/2607.27066)

[Overview](#overview) · [Highlights](#highlights) · [Method](#method-at-a-glance) · [Installation](#installation) · [Data](#data) · [Training](#training) · [Inference](#inference) · [Citation](#citation)

---

Scientific figure assessment in peer review is not natural-image IQA: a figure must be legible, support the manuscript’s claims, and present a clear visual hierarchy.

**SciFigAlign** fine-tunes CLIP Vision and SciBERT end-to-end so that each figure crop is scored against its caption, citing paragraphs, and light paper context on four peer-review dimensions (1–5): Clarity, Relevance, Informativeness, and Structure.

**Release scope.** This repository contains training / inference code, paper hyperparameters, and a **300-figure human-rated demo** with splits and PNG crops. The full 3,857-figure corpus is on Hugging Face: [haihanlamu/SciFigAlign](https://huggingface.co/datasets/haihanlamu/SciFigAlign).

<p align="center">
  <img src="assets/img/fig1.png" alt="SciFigAlign overview" width="92%" />
</p>
<p align="center"><em>Figure 1. Prior paradigms vs. SciFigAlign: manuscript-grounded CL / RE / IN / ST scoring.</em></p>

---

## Highlights

- **Manuscript-grounded inputs.** Figure crop + caption + citing context + light paper metadata, not isolated pixels.
- **Four-dimensional rubric.** Clarity (CL), Relevance (RE), Informativeness (IN), Structure (ST) on a 1–5 scale.
- **Fine-tuned multimodal scorer.** CLIP ViT-B/32 + SciBERT, per-modality cross-attention, CubeMLP fusion.
- **Joint objective.** SmoothL1 regression + within-paper ranking hinge (λ=0.2, margin=1.0, τ=0.5).
- **Strong test result.** Macro MAE **0.3524**, within-paper pairwise accuracy **81.64%** on the human-rated test set (*n*=396) — ≈59% relative MAE reduction vs. the best LLM-as-judge baseline (MAE 0.864).

---

## Overview

| Item | Value |
|------|-------|
| Labeled figures | 3,857 |
| Source papers | 3,126 (ICLR / NeurIPS / ICML) |
| Human-rated test set | 396 (paper-level split) |
| Rubric dimensions | CL, RE, IN, ST (1–5) |
| Encoders | CLIP ViT-B/32 + SciBERT |
| Test MAE / PA / SRCC | 0.3524 / 81.64% / 0.3088 |

---

## Method at a Glance

<p align="center">
  <img src="assets/img/fig2.png" alt="SciFigAlign architecture" width="92%" />
</p>
<p align="center"><em>Figure 2. Five streams → text→vision cross-attention → CubeMLP → CL / RE / IN / ST heads.</em></p>

| Stage | Role |
|-------|------|
| **Inputs** | Figure crop, caption, citing paragraphs (denoised), abstract / section cues, metadata |
| **Encoders** | CLIP vision + SciBERT text, fine-tuned end-to-end |
| **Alignment** | Per-modality cross-attention (text queries visual patches) |
| **Fusion** | CubeMLP over modality tokens |
| **Heads** | Four dimension-specific regressors |
| **Loss** | SmoothL1 + within-paper ranking hinge |

Paper defaults live in [`config/paper_config.json`](config/paper_config.json).

### Rubric

| Dim | Name | Focus |
|-----|------|-------|
| **CL** | Clarity | Print-scale legibility, contrast, labels |
| **RE** | Relevance | Fit to caption, cites, and narrative |
| **IN** | Informativeness | Scientific content for the claimed result |
| **ST** | Structure | Panel hierarchy and reading path |

### Corpus snapshot

<p align="center">
  <img src="assets/img/fig3.png" alt="SciFigAlign corpus statistics" width="92%" />
</p>
<p align="center"><em>Figure 3. Rubric profiles, type mix, score density, and construction funnel.</em></p>

---

## Repository Layout

```
.
├── train.py                 # training entry (SmoothL1 + ranking)
├── predict.py               # checkpoint inference → JSONL
├── prepare_dataset.py       # build training JSONL from raw exports
├── split_dataset.py         # paper-level 80/10/10 splits
├── config/paper_config.json # paper hyperparameters & stats
├── src/
│   ├── models/model.py      # CLIP + SciBERT + CrossAttn + CubeMLP
│   ├── dataset.py
│   ├── context_denoise.py
│   └── utils/
├── scripts/
│   ├── eval_pairwise_accuracy.py
│   └── run_appendix_scoring_demo.py
├── data/                    # 300-figure demo corpus
│   ├── figures/
│   ├── splits/{train,val,test}.jsonl
│   └── demo_corpus_meta.json
└── assets/                  # paper figures & CSS
```

---

## Installation

Python **3.10+** is recommended. Clone and create a virtual environment:

```bash
git clone https://github.com/FrankDengAI/SciFigAlign.git
cd SciFigAlign
python -m venv .venv
```

Activate with `source .venv/bin/activate` (macOS / Linux) or `.venv\Scripts\Activate.ps1` (Windows PowerShell), then:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
```

The first run downloads SciBERT and CLIP weights from Hugging Face (`allenai/scibert_scivocab_uncased`, `openai/clip-vit-base-patch32`).

---

## Data

### Demo pack (included)

This repo ships a **300-figure** human-rated demo under `data/`:

```
data/
├── figures/                 # PNG crops
├── splits/
│   ├── train.jsonl          # 240
│   ├── val.jsonl            # 30
│   └── test.jsonl           # 30
└── demo_corpus_meta.json
```

Paths inside the JSONL files are resolved via [`src/path_config.py`](src/path_config.py) against `data/figures/`.

### Full corpus

Complete 3,857-figure release: [huggingface.co/datasets/haihanlamu/SciFigAlign](https://huggingface.co/datasets/haihanlamu/SciFigAlign)

```bash
git lfs install
git clone https://huggingface.co/datasets/haihanlamu/SciFigAlign
```

Place images under `./data/figures`, `./data_source/figures`, or `../capstone-figures`, then rebuild JSONL with `prepare_dataset.py` / `split_dataset.py` as needed. Paper protocol: **paper-level** 80/10/10 split, seed **42**.

---

## Training

Demo training (paper-aligned ranking loss):

```bash
python train.py \
  --train_jsonl data/splits/train.jsonl \
  --val_jsonl data/splits/val.jsonl \
  --output_dir runs/demo \
  --verify_images \
  --use_ranking_loss \
  --ranking_weight 0.2 \
  --epochs 3
```

Key defaults match the paper: AdamW lr `2e-5`, batch size `4`, SmoothL1, ranking λ=`0.2`, margin=`1.0`, min-diff τ=`0.5`, seed `42`. Citing-context denoising is on by default (`src/context_denoise.py`); pass `--no_denoise_context` to disable.

---

## Inference

```bash
python predict.py \
  --checkpoint runs/demo/best.pt \
  --input_jsonl data/splits/test.jsonl \
  --output_jsonl outputs/test_predictions.jsonl \
  --verify_images
```

### Pairwise accuracy

```bash
python scripts/eval_pairwise_accuracy.py \
  --predictions outputs/test_predictions.jsonl
```

Within-paper pairs with gold gap ≥ 0.5 are counted (paper τ=0.5).

---

## Reproducibility Notes

- Entry points take explicit paths; no machine-local usernames are required for the demo.
- Seeds default to **42** for splitting and training.
- Exact paper numbers additionally require the full corpus, paper-matched manifests, and a matched software stack (record resolved `torch` / `transformers` versions).
- Checkpoints are not shipped in this repository; train from the demo or full corpus.

---

## License

Source code is released under the **MIT License**. The Hugging Face corpus ([haihanlamu/SciFigAlign](https://huggingface.co/datasets/haihanlamu/SciFigAlign)) is released under **Apache-2.0**.

---

## Citation

If you use SciFigAlign, please cite:

```bibtex
@article{xu2026scifigalign,
  title={SciFigAlign: Scoring Scientific Figures by Fine-tuned Alignment of Visuals with Manuscript Evidence},
  author={Xu, Chuanzhi and Deng, Zihan and Liang, Huiqi and Yue, Chengkun and Cui, Zhanlin and Ye, Pengfei and Cai, Weidong},
  journal={arXiv preprint arXiv:2607.27066},
  year={2026}
}
```
