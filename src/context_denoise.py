"""Citing-context denoising used by the deployed SciFigAlign checkpoint (Appendix C)."""
from __future__ import annotations

import re
from typing import Any, Dict, List


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\[[0-9,\s]+\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_paragraphs(value: Any) -> List[str]:
    if value is None:
        return []
    paragraphs = value if isinstance(value, list) else [value]
    cleaned: List[str] = []
    for p in paragraphs:
        if isinstance(p, dict):
            p = " ".join(str(v) for v in p.values())
        p = clean_text(p)
        if len(p) >= 20:
            cleaned.append(p)
    return cleaned


def get_caption_words(row: Dict[str, Any]) -> set[str]:
    caption = clean_text(row.get("caption_text", ""))
    return set(re.findall(r"[a-zA-Z]{3,}", caption.lower()))


def paragraph_score(paragraph: str, caption_words: set[str]) -> int:
    p_lower = paragraph.lower()
    words = set(re.findall(r"[a-zA-Z]{3,}", p_lower))
    overlap_score = len(words & caption_words)
    figure_score = 3 if re.search(r"\b(fig|figure|panel|plot|chart|graph|table)\b", p_lower) else 0
    length_score = 1 if 50 <= len(paragraph) <= 900 else 0
    return overlap_score + figure_score + length_score


def select_relevant_paragraphs(
    row: Dict[str, Any],
    field_name: str,
    *,
    max_items: int = 3,
    max_chars: int = 1200,
) -> List[str]:
    paragraphs = normalize_paragraphs(row.get(field_name, []))
    if not paragraphs:
        return []
    caption_words = get_caption_words(row)
    scored = sorted(
        ((paragraph_score(p, caption_words), p) for p in paragraphs),
        key=lambda x: x[0],
        reverse=True,
    )
    selected: List[str] = []
    total_chars = 0
    for _, p in scored:
        if total_chars + len(p) > max_chars:
            continue
        selected.append(p)
        total_chars += len(p)
        if len(selected) >= max_items:
            break
    return selected


def denoise_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Deployed full-denoised input: caption + citing (+ nearby), no abstract/section."""
    out = dict(row)
    out["caption_text"] = clean_text(out.get("caption_text", ""))
    out["citing_paragraphs"] = select_relevant_paragraphs(out, "citing_paragraphs")
    out["nearby_context"] = select_relevant_paragraphs(out, "nearby_context", max_items=2, max_chars=800)
    out["abstract"] = ""
    out["section_title"] = ""
    return out
