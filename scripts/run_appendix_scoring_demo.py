#!/usr/bin/env python3
"""Score ~20 figures for appendix case study + export explainability artifacts.

SciFigAlign outputs per-dimension scores (CL/RE/IN/ST) plus modality weights and
cross-attention patch focus — not free-text rationales.

Usage (from code/):
  python scripts/run_appendix_scoring_demo.py
  python scripts/run_appendix_scoring_demo.py --checkpoint ./runs/run1/best.pt --skip_train
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

# Allow imports from code/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402

DIMENSIONS = ["clarity", "relevance", "informativeness", "structure"]
DIM_LABELS = {"clarity": "CL", "relevance": "RE", "informativeness": "IN", "structure": "ST"}
MODALITIES = ["image", "caption", "context", "abstract", "meta"]
DEFAULT_EVAL1200 = ROOT.parent / "1. SciFigQual-Bench" / "datasets" / "eval1200"
DEFAULT_OUTPUT = ROOT.parent / "AAAI2027" / "paper" / "appendix_scoring_case_study"
OVERVIEW_JSON = ROOT / "frontend" / "public" / "dataset_overview.json"
FIGURES_HF_DIR = ROOT / "data_source" / "figures" / "figures"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval1200_dir", type=Path, default=DEFAULT_EVAL1200)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "run1" / "best.pt")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--prepare_only", action="store_true", help="Select samples + write input JSONL only")
    p.add_argument("--train_epochs", type=int, default=3)
    p.add_argument("--download_figures", action="store_true", help="Download capstone-figures from HF if missing")
    return p.parse_args()


def find_checkpoint(path: Path) -> Optional[Path]:
    candidates = [
        path,
        ROOT / "runs" / "run1" / "best.pt",
        ROOT / "run_result" / "model" / "best.pt",
    ]
    env = os.environ.get("SCIFIGALIGN_CHECKPOINT")
    if env:
        candidates.insert(0, Path(env))
    for c in candidates:
        if c and c.exists():
            return c.resolve()
    return None


def download_capstone_figures():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download
    target = ROOT / "data_source" / "figures"
    target.mkdir(parents=True, exist_ok=True)
    print(f"[download] HuggingFace anonymous-corpus/capstone-figures -> {target}")
    snapshot_download(
        repo_id="anonymous-corpus/capstone-figures",
        repo_type="dataset",
        local_dir=str(target),
        allow_patterns=["figures/*"],
    )


def overview_to_scifigalign_jsonl(out_path: Path, figures_root: Path) -> int:
    rows_raw = json.loads(OVERVIEW_JSON.read_text(encoding="utf-8"))
    out_rows: List[Dict[str, Any]] = []
    missing = 0
    for r in rows_raw:
        rel = r.get("image_path", "")
        img = figures_root / rel.replace("/", os.sep)
        if not img.exists():
            img = figures_root.parent / rel.replace("/", os.sep)
        if not img.exists():
            missing += 1
            continue
        scores = {}
        for dim in DIMENSIONS:
            scores[dim] = {
                "score": float(r.get(dim, 0)),
                "rationale": r.get(f"{dim}_rationale", "") or "",
            }
        out_rows.append({
            "paper_id": r.get("paper_id", ""),
            "figure_id": r.get("figure_id", ""),
            "paper_title": r.get("venue", ""),
            "figure_type": r.get("figure_type", ""),
            "figure_image_path": str(img.resolve()),
            "caption_text": r.get("caption", "") or "",
            "citing_paragraphs": [],
            "abstract": "",
            "scores": scores,
            "label_source": r.get("source", "human"),
        })
    write_jsonl(str(out_path), out_rows)
    print(f"[jsonl] wrote {len(out_rows)} training rows ({missing} missing images) -> {out_path}")
    return len(out_rows)


def split_train_val(rows: List[Dict], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List, List]:
    rng = random.Random(seed)
    papers = sorted({r["paper_id"] for r in rows})
    rng.shuffle(papers)
    n_val = max(1, int(len(papers) * val_ratio))
    val_papers = set(papers[:n_val])
    train, val = [], []
    for r in rows:
        (val if r["paper_id"] in val_papers else train).append(r)
    return train, val


def train_checkpoint(train_jsonl: Path, val_jsonl: Path, output_dir: Path, epochs: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--train_jsonl", str(train_jsonl),
        "--val_jsonl", str(val_jsonl),
        "--output_dir", str(output_dir),
        "--epochs", str(epochs),
        "--batch_size", "4",
        "--verify_images",
        "--use_ranking_loss",
        "--ranking_weight", "0.2",
    ]
    print("[train]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def eval1200_row_to_scifigalign(row: Dict, images_dir: Path) -> Dict:
    img_rel = row.get("image", "")
    img_path = (images_dir / Path(img_rel).name).resolve()
    if not img_path.exists():
        img_path = (images_dir.parent / img_rel).resolve()
    context = row.get("context_texts") or []
    meta_bits = [row.get("title", ""), row.get("venue", ""), str(row.get("year", "")), row.get("section", "")]
    meta_text = " | ".join(x for x in meta_bits if x)
    return {
        "paper_id": row.get("paper_id", ""),
        "figure_id": row.get("figure_id", ""),
        "paper_title": row.get("title", ""),
        "figure_type": row.get("domain_l2", "") or row.get("domain_l1", ""),
        "figure_image_path": str(img_path),
        "caption_text": row.get("caption", "") or "",
        "citing_paragraphs": context[:3],
        "abstract": row.get("title", "") or "",
        "section_title": row.get("section", "") or "",
        "scores": {},
        "label_source": "eval1200",
        # reference human subscores (SciFigQual rubric, 1-10) for appendix comparison
        "eval1200_human": {
            "visual_clarity": row.get("human_visual_clarity"),
            "structure_layout": row.get("human_structure_layout"),
            "caption_consistency": row.get("human_caption_consistency"),
            "context_consistency": row.get("human_context_consistency"),
            "misleading_risk": row.get("human_misleading_risk"),
            "overall": row.get("human_overall_score"),
        },
    }


def select_eval1200_samples(figures_jsonl: Path, images_dir: Path, n: int, seed: int) -> List[Dict]:
    rows = read_jsonl(str(figures_jsonl))
    # keep rows with existing images; prefer diversity by venue + score spread
    valid = []
    for r in rows:
        conv = eval1200_row_to_scifigalign(r, images_dir)
        if Path(conv["figure_image_path"]).exists():
            valid.append((r, conv))
    rng = random.Random(seed)
    by_venue: Dict[str, list] = {}
    for r, conv in valid:
        by_venue.setdefault(r.get("venue", "?"), []).append((r, conv))
    picked: List[Tuple[Dict, Dict]] = []
    venues = sorted(by_venue.keys())
    rng.shuffle(venues)
    idx = 0
    while len(picked) < n and venues:
        v = venues[idx % len(venues)]
        pool = by_venue[v]
        if pool:
            choice = rng.choice(pool)
            if choice not in picked:
                picked.append(choice)
            by_venue[v].remove(choice)
            if not by_venue[v]:
                venues.remove(v)
        idx += 1
        if idx > n * 20:
            break
    if len(picked) < n:
        rest = [x for x in valid if x not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return [conv for _, conv in picked[:n]]


def run_predict(checkpoint: Path, input_jsonl: Path, output_jsonl: Path):
    cmd = [
        sys.executable, str(ROOT / "predict.py"),
        "--checkpoint", str(checkpoint),
        "--input_jsonl", str(input_jsonl),
        "--output_jsonl", str(output_jsonl),
        "--batch_size", "4",
        "--verify_images",
    ]
    print("[predict]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def patch_focus_to_heatmap(patch_focus: List[float], img: Image.Image) -> np.ndarray:
    n = len(patch_focus)
    side = int(round(math.sqrt(n)))
    while side * side < n:
        side += 1
    arr = np.array(patch_focus[: side * side], dtype=np.float32)
    if arr.size < side * side:
        arr = np.pad(arr, (0, side * side - arr.size))
    grid = arr.reshape(side, side)
    grid = grid - grid.min()
    if grid.max() > 0:
        grid = grid / grid.max()
    heat = Image.fromarray((grid * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    return np.array(heat) / 255.0


def render_case_card(row: Dict, out_path: Path):
    img_path = Path(row["figure_image_path"])
    img = Image.open(img_path).convert("RGB")
    scores = row["pred_scores"]
    expl = row.get("explanation", {})
    dim_mod = expl.get("dimension_modality_importance", {})
    global_mod = expl.get("global_modality_importance", {})
    patch_focus = expl.get("cross_attention_patch_focus", {})
    # use caption modality patch focus if available
    pf = patch_focus.get("caption") or patch_focus.get("context") or next(iter(patch_focus.values()), [])

    fig = plt.figure(figsize=(10, 6), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], wspace=0.25, hspace=0.35)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_attn = fig.add_subplot(gs[0, 1])
    ax_scores = fig.add_subplot(gs[1, 0])
    ax_mod = fig.add_subplot(gs[1, 1])

    ax_img.imshow(img)
    ax_img.set_title(row.get("figure_id", ""), fontsize=9)
    ax_img.axis("off")

    if pf:
        heat = patch_focus_to_heatmap(pf, img)
        ax_attn.imshow(img)
        ax_attn.imshow(heat, cmap="magma", alpha=0.45)
        ax_attn.set_title("Cross-attn patch focus", fontsize=9)
    else:
        ax_attn.text(0.5, 0.5, "No attention export", ha="center", va="center")
    ax_attn.axis("off")

    vals = [scores[d] for d in DIMENSIONS]
    colors = ["#0a6b6f", "#245a85", "#9a7200", "#c04e1a"]
    ax_scores.bar([DIM_LABELS[d] for d in DIMENSIONS], vals, color=colors)
    ax_scores.set_ylim(1, 5)
    ax_scores.set_ylabel("Score (1–5)")
    ax_scores.set_title("SciFigAlign predictions", fontsize=9)
    for i, v in enumerate(vals):
        ax_scores.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)

    # modality weights for clarity vs relevance as example
    if dim_mod:
        x = np.arange(len(MODALITIES))
        w_cl = [dim_mod.get("clarity", {}).get(m, 0) for m in MODALITIES]
        w_re = [dim_mod.get("relevance", {}).get(m, 0) for m in MODALITIES]
        ax_mod.bar(x - 0.18, w_cl, width=0.36, label="CL", color="#0a6b6f")
        ax_mod.bar(x + 0.18, w_re, width=0.36, label="RE", color="#245a85")
        ax_mod.set_xticks(x)
        ax_mod.set_xticklabels(MODALITIES, rotation=20, ha="right", fontsize=8)
        ax_mod.set_title("Modality weights (CL vs RE)", fontsize=9)
        ax_mod.legend(fontsize=8)
    elif global_mod:
        ax_mod.bar(global_mod.keys(), global_mod.values(), color="#5c6f82")
        ax_mod.set_title("Global modality weights", fontsize=9)
        ax_mod.tick_params(axis="x", rotation=20)

    cap = row.get("caption_text", "")[:120]
    fig.suptitle(cap + ("..." if len(row.get("caption_text", "")) > 120 else ""), fontsize=8, y=0.98)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_summary_grid(rows: List[Dict], out_path: Path, max_show: int = 6):
    show = rows[:max_show]
    n = len(show)
    cols = 3
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(12, 3.6 * rows_n), dpi=150)
    if rows_n == 1:
        axes = np.array([axes])
    axes = axes.reshape(rows_n, cols)
    for ax in axes.flat:
        ax.axis("off")
    for i, row in enumerate(show):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        img = Image.open(row["figure_image_path"]).convert("RGB")
        ax.imshow(img)
        sc = row["pred_scores"]
        txt = "  ".join(f"{DIM_LABELS[d]}:{sc[d]:.1f}" for d in DIMENSIONS)
        ax.set_title(f"{row.get('figure_id','')}\n{txt}", fontsize=8)
        ax.axis("off")
    fig.suptitle("SciFigAlign scoring examples (eval1200 subset)", fontsize=11)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_process_diagram(out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 2.8), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.2, 1.2, "Figure\n+ caption\n+ context"),
        (2.3, 1.2, "CLIP +\nSciBERT"),
        (4.3, 1.2, "Cross-attn\n+ CubeMLP"),
        (6.3, 1.2, "4 score\nheads"),
        (8.2, 1.2, "CL / RE / IN / ST\n+ modality weights\n+ patch focus"),
    ]
    for i, (x, y, label) in enumerate(boxes):
        rect = mpatches.FancyBboxPatch(
            (x, y), 1.6, 1.0, boxstyle="round,pad=0.02", linewidth=1,
            edgecolor="#1b2838", facecolor="#e5eff8" if i % 2 == 0 else "#e3f3f4",
        )
        ax.add_patch(rect)
        ax.text(x + 0.8, y + 0.5, label, ha="center", va="center", fontsize=8)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 1.75, y + 0.5), xytext=(x + 1.95, y + 0.5),
                        arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(5, 2.6, "SciFigAlign inference outputs scores and explainability signals (not free-text rationales)", ha="center", fontsize=9)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.eval1200_dir / "images"
    figures_jsonl = args.eval1200_dir / "figures.jsonl"
    if not figures_jsonl.exists():
        raise FileNotFoundError(f"Missing eval1200 metadata: {figures_jsonl}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing eval1200 images: {images_dir}")

    selected = select_eval1200_samples(figures_jsonl, images_dir, args.n_samples, args.seed)
    input_jsonl = args.output_dir / "eval20_input.jsonl"

    images_out = args.output_dir / "images"
    images_out.mkdir(exist_ok=True)
    for row in selected:
        src = Path(row["figure_image_path"])
        dst = images_out / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        row["figure_image_path"] = str(dst.resolve())

    write_jsonl(str(input_jsonl), selected)
    render_process_diagram(args.output_dir / "AppFig_scoring_process.pdf")

    readme = """# SciFigAlign Appendix Scoring Case Study

## What SciFigAlign outputs
- **Per-dimension scores (1–5):** clarity, relevance, informativeness, structure
- **Explainability (no free-text reasons):**
  - global / per-dimension modality weights (image, caption, context, abstract, meta)
  - cross-attention patch focus vectors (can overlay as heatmap)

SciFigAlign does **not** emit natural-language rationales (unlike LLM judge or eval1200 human annotations).

## Run scoring (requires trained checkpoint)
```bash
cd code
python scripts/run_appendix_scoring_demo.py \\
  --skip_train \\
  --checkpoint ./runs/run1/best.pt \\
  --output_dir ../AAAI2027/paper/appendix_scoring_case_study
```

Place `best.pt` at `code/runs/run1/best.pt` (from Google Drive or your training run).
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")

    if args.prepare_only:
        write_json(
            str(args.output_dir / "manifest_prepare.json"),
            {"n_samples": len(selected), "input_jsonl": str(input_jsonl), "status": "prepared"},
        )
        print(f"[prepare_only] {len(selected)} samples -> {args.output_dir}")
        return

    ckpt = find_checkpoint(args.checkpoint)
    if ckpt is None and not args.skip_train:
        if args.download_figures or not FIGURES_HF_DIR.exists():
            download_capstone_figures()
        splits_dir = ROOT / "data" / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        merged = splits_dir / "from_overview.jsonl"
        n_ok = overview_to_scifigalign_jsonl(merged, FIGURES_HF_DIR)
        if n_ok < 500:
            raise RuntimeError(
                f"Only {n_ok} labeled images found under {FIGURES_HF_DIR}. "
                "Run with --download_figures or place capstone-figures under code/data_source/figures/"
            )
        rows = read_jsonl(str(merged))
        train, val = split_train_val(rows)
        train_p, val_p = splits_dir / "train_demo.jsonl", splits_dir / "val_demo.jsonl"
        write_jsonl(str(train_p), train)
        write_jsonl(str(val_p), val)
        run_dir = ROOT / "runs" / "run1"
        train_checkpoint(train_p, val_p, run_dir, args.train_epochs)
        ckpt = find_checkpoint(run_dir / "best.pt")
    if ckpt is None:
        raise FileNotFoundError(
            "No checkpoint found. Use --prepare_only to prepare inputs, or pass --checkpoint path/to/best.pt"
        )
    print(f"[checkpoint] {ckpt}")

    pred_jsonl = args.output_dir / "predictions.jsonl"
    run_predict(ckpt, input_jsonl, pred_jsonl)
    preds = read_jsonl(str(pred_jsonl))

    cards_dir = args.output_dir / "case_cards"
    cards_dir.mkdir(exist_ok=True)
    for row in preds:
        out = cards_dir / f"{row['figure_id']}.png"
        render_case_card(row, out)

    render_summary_grid(preds, args.output_dir / "AppFig_scoring_grid.pdf", max_show=6)
    render_process_diagram(args.output_dir / "AppFig_scoring_process.pdf")
    # copy best 2 case cards for direct appendix use
    if len(preds) >= 2:
        shutil.copy(cards_dir / f"{preds[0]['figure_id']}.png", args.output_dir / "AppFig_scoring_example1.png")
        shutil.copy(cards_dir / f"{preds[1]['figure_id']}.png", args.output_dir / "AppFig_scoring_example2.png")

    summary = {
        "n_samples": len(preds),
        "checkpoint": str(ckpt),
        "output_schema": {
            "pred_scores": "clarity/relevance/informativeness/structure (1-5)",
            "explanation": "global_modality_importance, dimension_modality_importance, cross_attention_patch_focus",
            "note": "SciFigAlign does not emit natural-language rationales; eval1200 human reasons are in eval1200_human field of input only.",
        },
        "figures": [str(cards_dir / f"{r['figure_id']}.png") for r in preds],
    }
    write_json(str(args.output_dir / "manifest.json"), summary)
    print(f"[done] outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
