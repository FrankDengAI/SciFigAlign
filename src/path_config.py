"""Resolve figure/image paths for SciFigAlign (paper-aligned layout)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

# code/ directory
CODE_ROOT = Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    for p in [CODE_ROOT.parent, *CODE_ROOT.parents]:
        if (p / "capstone-figures").exists():
            return p
    return CODE_ROOT.parent


PROJECT_ROOT = _repo_root()

FIGURE_SEARCH_ROOTS: tuple[Path, ...] = (
    CODE_ROOT / "data" / "figures",
    CODE_ROOT / "data_source" / "figures",
    PROJECT_ROOT / "capstone-figures",
    CODE_ROOT.parent / "data" / "figures",
)


def normalize_existing_path(path: str) -> str:
    return path.replace("\\", os.sep).replace("/", os.sep)


def resolve_figure_path(
    path: str,
    *,
    extra_roots: Optional[Iterable[Path]] = None,
    must_exist: bool = False,
) -> str:
    """Resolve a figure path to an absolute path under known project roots."""
    if not path:
        return path

    raw = normalize_existing_path(path)
    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate.resolve())

    rel = raw.lstrip("./\\")
    rel_posix = rel.replace(os.sep, "/")

    roots = list(FIGURE_SEARCH_ROOTS)
    if extra_roots:
        roots = list(extra_roots) + roots

    # Direct join under each root
    for root in roots:
        for suffix in (rel_posix, rel_posix.replace("figures/", "", 1) if rel_posix.startswith("figures/") else rel_posix):
            p = (root / suffix.replace("/", os.sep))
            if p.exists():
                return str(p.resolve())

    # Search by basename (portable supplementary zips)
    basename = Path(rel_posix).name
    for root in roots:
        matches = list(root.rglob(basename))
        if matches:
            return str(matches[0].resolve())

    if must_exist:
        raise FileNotFoundError(f"Figure not found: {path}")
    return str((CODE_ROOT / rel).resolve())


def to_relative_figure_path(abs_path: str, base_dir: Path) -> str:
    """Store portable relative paths inside supplementary JSONL."""
    abs_p = Path(abs_path).resolve()
    base = base_dir.resolve()
    try:
        return abs_p.relative_to(base).as_posix()
    except ValueError:
        return abs_p.name
