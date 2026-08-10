"""Lightweight helpers for exporting training monitor tables."""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def flatten_metrics(metrics: Mapping[str, Any], *, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested metric dicts into dotted keys."""
    flat: Dict[str, Any] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_metrics(value, prefix=name))
        else:
            flat[name] = value
    return flat


def write_json_file(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv_file(path: str, rows: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    materialised: List[Mapping[str, Any]] = list(rows)
    if not materialised:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in materialised:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in materialised:
            writer.writerow(dict(row))
