#!/usr/bin/env python3
"""Compute within-paper pairwise accuracy (PA) from prediction JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

DIMENSIONS = ["clarity", "relevance", "informativeness", "structure"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="JSONL with scores + predictions")
    p.add_argument("--tau", type=float, default=0.5, help="Minimum gold overall gap")
    p.add_argument("--pred_field", default="predicted_scores")
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def overall_score(row: Dict, field: str) -> float:
    if field == "gold":
        scores = row.get("scores", {})
        vals = []
        for dim in DIMENSIONS:
            v = scores.get(dim, 0)
            if isinstance(v, dict):
                v = v.get("score", 0)
            vals.append(float(v))
    else:
        pred = row.get(field) or row.get("predictions") or {}
        vals = [float(pred.get(dim, pred.get(dim.upper(), 0))) for dim in DIMENSIONS]
    return sum(vals) / len(vals)


def pairwise_accuracy(rows: List[Dict], tau: float, pred_field: str) -> Tuple[float, int]:
    by_paper: Dict[str, List[Tuple[float, float]]] = {}
    for row in rows:
        pid = row.get("paper_id", "")
        by_paper.setdefault(pid, []).append(
            (overall_score(row, "gold"), overall_score(row, pred_field))
        )

    correct = total = 0
    for items in by_paper.values():
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                gy_i, gy_j = items[i][0], items[j][0]
                if abs(gy_i - gy_j) < tau:
                    continue
                gp_i, gp_j = items[i][1], items[j][1]
                if (gy_i - gy_j) * (gp_i - gp_j) > 0:
                    correct += 1
                total += 1
    pa = correct / total if total else 0.0
    return pa, total


def main():
    args = parse_args()
    rows = read_jsonl(Path(args.predictions))
    pa, n_pairs = pairwise_accuracy(rows, args.tau, args.pred_field)
    print(json.dumps({"pairwise_accuracy": pa, "n_pairs": n_pairs, "tau": args.tau}, indent=2))


if __name__ == "__main__":
    main()
