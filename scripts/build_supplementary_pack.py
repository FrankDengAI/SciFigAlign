#!/usr/bin/env python3
"""Build SciFigAlign supplementary material zip (<50 MB by default).

Packages:
  - Core training/inference code aligned with the AAAI paper
  - Demo corpus JSONL + copied figure PNGs (human-rated subset)
  - Appendix case-study images (20 figures) + eval20_input.jsonl
  - paper_config.json + README

Usage (from code/):
  python scripts/build_supplementary_pack.py
  python scripts/build_supplementary_pack.py --max_zip_mb 50 --max_demo_figures 100
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]


def find_repo_root(start: Path) -> Path:
    """Locate SciFigAlign repo root (directory that contains capstone-figures/)."""
    for p in [start, *start.parents]:
        if (p / "capstone-figures").exists():
            return p
    return start.parent if start.name == "code" else start


REPO_ROOT = find_repo_root(ROOT.parent if ROOT.name == "code" else ROOT)
PROJECT_ROOT = REPO_ROOT
OVERVIEW_JSON = ROOT / "frontend" / "public" / "dataset_overview.json"
if not OVERVIEW_JSON.exists():
    _fallback = REPO_ROOT / "docs" / "CS64-1.zip"
    if _fallback.exists():
        import zipfile
        OVERVIEW_JSON.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(_fallback) as zf:
            OVERVIEW_JSON.write_bytes(zf.read("CS64-1/frontend/public/dataset_overview.json"))
CASE_STUDY_DIR = REPO_ROOT / "AAAI2027" / "paper" / "appendix_scoring_case_study"
CAPSTONE_ROOT = REPO_ROOT / "capstone-figures"

CODE_FILES = [
    "train.py",
    "predict.py",
    "split_dataset.py",
    "prepare_dataset.py",
    "requirements-core.txt",
    "config/paper_config.json",
    "src/__init__.py",
    "src/dataset.py",
    "src/path_config.py",
    "src/context_denoise.py",
    "src/models/model.py",
    "src/utils/io.py",
    "src/utils/metrics.py",
    "src/utils/seed.py",
    "src/utils/device.py",
    "scripts/build_supplementary_pack.py",
    "scripts/run_appendix_scoring_demo.py",
    "scripts/eval_pairwise_accuracy.py",
]

DIMENSIONS = ["clarity", "relevance", "informativeness", "structure"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_zip", type=Path, default=REPO_ROOT / "SciFigAlign_supplementary.zip")
    p.add_argument("--staging_dir", type=Path, default=REPO_ROOT / "SciFigAlign_supplementary")
    p.add_argument("--max_zip_mb", type=float, default=50.0)
    p.add_argument("--max_demo_figures", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep_staging", action="store_true")
    return p.parse_args()


def load_overview() -> List[Dict[str, Any]]:
    return json.loads(OVERVIEW_JSON.read_text(encoding="utf-8"))


def resolve_capstone_image(rel_path: str) -> Path | None:
    rel = rel_path.replace("\\", "/").lstrip("./")
    for base in (CAPSTONE_ROOT, ROOT / "data_source" / "figures"):
        for candidate in (base / rel, base / rel.replace("figures/", "", 1)):
            if candidate.exists():
                return candidate.resolve()
    return None


def overview_row_to_jsonl(row: Dict[str, Any], rel_image_path: str) -> Dict[str, Any]:
    scores = {
        dim: {"score": float(row.get(dim, 0)), "rationale": row.get(f"{dim}_rationale", "") or ""}
        for dim in DIMENSIONS
    }
    return {
        "paper_id": row.get("paper_id", ""),
        "figure_id": row.get("figure_id", ""),
        "paper_title": row.get("venue", ""),
        "venue": row.get("venue", ""),
        "figure_type": row.get("figure_type", ""),
        "figure_image_path": rel_image_path,
        "caption_text": row.get("caption", "") or "",
        "citing_paragraphs": [],
        "abstract": "",
        "section_title": "",
        "nearby_context": [],
        "scores": scores,
        "label_source": row.get("source", "human"),
    }


def select_demo_rows(rows: List[Dict[str, Any]], max_figures: int, seed: int) -> List[Tuple[Dict[str, Any], Path, int]]:
    """Pick human-rated figures with existing images; prefer smaller PNGs for zip budget."""
    import random

    rng = random.Random(seed)
    candidates: List[Tuple[Dict[str, Any], Path, int]] = []
    for row in rows:
        if row.get("source") != "human":
            continue
        img = resolve_capstone_image(row.get("image_path", ""))
        if img is None:
            continue
        size = img.stat().st_size
        candidates.append((row, img, size))

    by_paper: Dict[str, List[Tuple[Dict[str, Any], Path, int]]] = defaultdict(list)
    for item in candidates:
        by_paper[item[0]["paper_id"]].append(item)

    paper_ids = list(by_paper.keys())
    rng.shuffle(paper_ids)

    picked: List[Tuple[Dict[str, Any], Path, int]] = []
    # Round-robin across papers for diversity, prefer smaller images within each paper
    while len(picked) < max_figures and paper_ids:
        next_papers = []
        for pid in paper_ids:
            pool = sorted(by_paper[pid], key=lambda x: x[2])
            if pool:
                picked.append(pool.pop(0))
                if pool:
                    next_papers.append(pid)
            if len(picked) >= max_figures:
                break
        paper_ids = next_papers

    return picked[:max_figures]


def paper_level_split(rows: List[Dict[str, Any]], seed: int = 42) -> Tuple[List, List, List]:
    import random

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["paper_id"]].append(row)
    paper_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(paper_ids)
    n = len(paper_ids)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_ids = set(paper_ids[:n_train])
    val_ids = set(paper_ids[n_train : n_train + n_val])
    test_ids = set(paper_ids[n_train + n_val :])
    train = [r for pid in train_ids for r in groups[pid]]
    val = [r for pid in val_ids for r in groups[pid]]
    test = [r for pid in test_ids for r in groups[pid]]
    return train, val, test


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_code(staging: Path) -> None:
    code_dst = staging / "code"
    code_dst.mkdir(parents=True, exist_ok=True)
    for rel in CODE_FILES:
        src = ROOT / rel
        if not src.exists():
            continue
        dst = code_dst / rel
        try:
            if src.resolve() == dst.resolve():
                continue
        except OSError:
            pass
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_case_study(staging: Path) -> None:
    dst = staging / "case_study"
    if not CASE_STUDY_DIR.exists():
        return
    if not dst.exists():
        shutil.copytree(CASE_STUDY_DIR, dst)
    else:
        # Refresh eval JSONL + images if case study source changed
        for name in ("eval20_input.jsonl",):
            src = CASE_STUDY_DIR / name
            if src.exists():
                shutil.copy2(src, dst / name)
        src_img = CASE_STUDY_DIR / "images"
        if src_img.exists():
            (dst / "images").mkdir(exist_ok=True)
            for img in src_img.glob("*.png"):
                shutil.copy2(img, dst / "images" / img.name)
    # Rewrite eval20 paths to portable relative paths
    eval_path = dst / "eval20_input.jsonl"
    if not eval_path.exists():
        return
    rows = []
    with eval_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    for row in rows:
        name = Path(row.get("figure_image_path", "")).name
        row["figure_image_path"] = f"images/{name}"
    write_jsonl(eval_path, rows)


def build_demo_corpus(staging: Path, max_figures: int, seed: int) -> int:
    overview = load_overview()
    picked = select_demo_rows(overview, max_figures, seed)
    fig_dst_root = staging / "data" / "figures"
    fig_dst_root.mkdir(parents=True, exist_ok=True)

    jsonl_rows: List[Dict[str, Any]] = []
    for row, src_img, _ in picked:
        dst_name = f"{row['paper_id']}__{row['figure_id']}.png"
        dst_path = fig_dst_root / dst_name
        shutil.copy2(src_img, dst_path)
        rel = f"figures/{dst_name}"
        jsonl_rows.append(overview_row_to_jsonl(row, rel))

    train, val, test = paper_level_split(jsonl_rows, seed=seed)
    splits_dir = staging / "data" / "splits"
    write_jsonl(splits_dir / "demo_all.jsonl", jsonl_rows)
    write_jsonl(splits_dir / "train.jsonl", train)
    write_jsonl(splits_dir / "val.jsonl", val)
    write_jsonl(splits_dir / "test.jsonl", test)

    meta = {
        "description": "Human-rated demo subset exported from dataset_overview.json",
        "n_figures": len(jsonl_rows),
        "n_papers": len({r['paper_id'] for r in jsonl_rows}),
        "split_protocol": "paper-level 80/10/10, seed=42",
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "note": "Full 3,857-figure corpus with citing paragraphs is released separately on HuggingFace.",
    }
    (staging / "data" / "demo_corpus_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return len(jsonl_rows)


def write_readme(staging: Path, n_demo: int) -> None:
    text = f"""# SciFigAlign Supplementary Material

This archive supports reproduction of the AAAI 2027 paper
**SciFigAlign: Scoring Scientific Figures by Fine-tuned Alignment of Visuals with Manuscript Evidence**.

## Contents

| Path | Description |
|------|-------------|
| `code/` | Training / inference code aligned with the paper (CLIP ViT-B/32 + SciBERT, CubeMLP, SmoothL1 + ranking) |
| `code/config/paper_config.json` | Hyperparameters and corpus statistics cited in the paper |
| `data/splits/` | Demo JSONL splits ({n_demo} human-rated figures, paper-level 80/10/10) |
| `data/figures/` | PNG crops for the demo subset |
| `case_study/` | 20 appendix scoring-case figures + `eval20_input.jsonl` |

## Quick start (demo train)

```bash
cd code
pip install -r requirements-core.txt
python train.py \\
  --train_jsonl ../data/splits/train.jsonl \\
  --val_jsonl ../data/splits/val.jsonl \\
  --output_dir ./runs/demo \\
  --verify_images \\
  --use_ranking_loss \\
  --ranking_weight 0.2 \\
  --epochs 3
python predict.py --checkpoint ./runs/demo/best.pt \\
  --input_jsonl ../data/splits/test.jsonl \\
  --output_jsonl ../data/splits/test_predictions.jsonl \\
  --verify_images
python scripts/eval_pairwise_accuracy.py \\
  --predictions ../data/splits/test_predictions.jsonl
```

## Full corpus

The complete 3,857-figure benchmark (334 MB PNGs) is on HuggingFace:
`anonymous-corpus/capstone-figures`.

Place images under `../capstone-figures/` or `code/data_source/figures/` and rebuild
JSONL with `prepare_dataset.py` / export scripts from the full repository.

## Paper-aligned defaults

- Paper-level split seed **42**
- AdamW lr **2e-5**, batch **4**, SmoothL1 + ranking λ=**0.2**, margin **1.0**, τ=**0.5**
- Deployed input: **full denoised** citing context (see `src/context_denoise.py`)
- Rubric dimensions: Clarity / Relevance / Informativeness / Structure (1–5)

## Appendix case study

```bash
python scripts/run_appendix_scoring_demo.py \\
  --output_dir ../case_study \\
  --prepare_only
```

Add a trained checkpoint via `--checkpoint path/to/best.pt` to score the 20 curated cases.
"""
    (staging / "README.md").write_text(text, encoding="utf-8")


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def make_zip(staging: Path, zip_path: Path) -> int:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file in sorted(staging.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(staging).as_posix())
    return zip_path.stat().st_size


def main() -> None:
    args = parse_args()
    if args.staging_dir.exists():
        # Refresh data/code inside existing supplementary folder without deleting case_study extras
        for sub in ("data",):
            target = args.staging_dir / sub
            if target.exists():
                shutil.rmtree(target)
    else:
        args.staging_dir.mkdir(parents=True)

    copy_code(args.staging_dir)
    copy_case_study(args.staging_dir)
    n_demo = build_demo_corpus(args.staging_dir, args.max_demo_figures, args.seed)
    write_readme(args.staging_dir, n_demo)

    staging_mb = dir_size_bytes(args.staging_dir) / (1024 * 1024)
    max_mb = args.max_zip_mb

    # Shrink demo set if staging exceeds budget (reserve ~5 MB for code/readme)
    while staging_mb > max_mb * 0.95 and args.max_demo_figures > 20:
        args.max_demo_figures = max(20, args.max_demo_figures - 20)
        shutil.rmtree(args.staging_dir / "data", ignore_errors=True)
        n_demo = build_demo_corpus(args.staging_dir, args.max_demo_figures, args.seed)
        write_readme(args.staging_dir, n_demo)
        staging_mb = dir_size_bytes(args.staging_dir) / (1024 * 1024)
        print(f"[resize] demo figures -> {args.max_demo_figures}, staging {staging_mb:.1f} MB")

    zip_bytes = make_zip(args.staging_dir, args.output_zip)
    zip_mb = zip_bytes / (1024 * 1024)

    print(f"[done] demo figures: {n_demo}")
    print(f"[done] staging: {staging_mb:.2f} MB -> {args.staging_dir}")
    print(f"[done] zip: {zip_mb:.2f} MB -> {args.output_zip}")

    if zip_mb > max_mb:
        print(f"[warn] zip exceeds {max_mb} MB budget; reduce --max_demo_figures")

    if not args.keep_staging and args.staging_dir.resolve() != REPO_ROOT / "SciFigAlign_supplementary":
        shutil.rmtree(args.staging_dir)


if __name__ == "__main__":
    main()
