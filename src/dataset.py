import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.context_denoise import denoise_row
from src.path_config import resolve_figure_path
from src.utils.io import read_jsonl

DIMENSIONS = ['clarity', 'relevance', 'informativeness', 'structure']


def normalize_existing_path(path: str) -> str:
    return path.replace('\\', os.sep).replace('/', os.sep)


class FigureQualityDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        verify_images: bool = False,
        include_weak_labels: bool = True,
        denoise_context: bool = True,
    ):
        rows = read_jsonl(jsonl_path)
        self.jsonl_dir = Path(jsonl_path).resolve().parent
        if not include_weak_labels:
            rows = [x for x in rows if x.get('label_source') != 'weak']
        if verify_images:
            filtered = []
            for row in rows:
                image_path = self._resolve_row_image_path(row)
                if not image_path or not Path(image_path).exists():
                    continue
                row['figure_image_path'] = image_path
                filtered.append(row)
            rows = filtered
        self.denoise_context = denoise_context
        self.rows = rows

    def _resolve_row_image_path(self, row: Dict[str, Any]) -> str:
        raw = row.get("figure_image_path", "")
        if not raw:
            return raw
        candidate = Path(raw)
        if candidate.is_absolute() and candidate.exists():
            return str(candidate.resolve())
        search_bases = [
            self.jsonl_dir,
            self.jsonl_dir.parent,
            self.jsonl_dir.parent / "data",
            self.jsonl_dir.parent.parent / "data",
        ]
        rel = raw.replace("\\", "/").lstrip("./")
        for base in search_bases:
            p = (base / rel.replace("/", os.sep))
            if p.exists():
                return str(p.resolve())
        return resolve_figure_path(raw)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        if self.denoise_context:
            row = denoise_row(row)
        image_path = self._resolve_row_image_path(row)
        image = Image.open(image_path).convert('RGB')
        caption = row.get('caption_text', '') or ''
        context = ' '.join(row.get('citing_paragraphs', [])[:3])
        abstract = row.get('abstract', '') or ''
        figure_type = row.get('figure_type', '') or ''
        title = row.get('paper_title', '') or ''
        decision = row.get('decision', '') or ''
        section_title = row.get('section_title', '') or ''
        nearby_context = ' '.join(row.get('nearby_context', [])[:2]) if isinstance(row.get('nearby_context'), list) else str(row.get('nearby_context', '') or '')

        labels = []
        scores = row.get('scores', {})
        for dim in DIMENSIONS:
            value = scores.get(dim, 0)
            if isinstance(value, dict):
                value = value.get('score', 0)
            labels.append(float(value))

        return {
            'image': image,
            'caption_text': caption,
            'context_text': context,
            'abstract_text': abstract,
            'meta_text': (
                f'Title: {title}. '
                f'Figure Type: {figure_type}. '
                f'Decision: {decision}. '
                f'Section: {section_title}. '
                f'Nearby: {nearby_context}'
            ),
            'labels': labels,
            'raw': row,
        }


class InferenceFigureDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        image = row['image'].convert('RGB')
        return {
            'image': image,
            'caption_text': row.get('caption_text', ''),
            'context_text': row.get('context_text', ''),
            'abstract_text': row.get('abstract_text', ''),
            'meta_text': row.get('meta_text', ''),
            'labels': [0.0, 0.0, 0.0, 0.0],
            'raw': row,
        }


class FigureCollator:
    def __init__(self, image_processor, tokenizer, max_text_length: int = 256):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def _encode_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt',
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [x['image'] for x in batch]
        captions = [x['caption_text'] for x in batch]
        contexts = [x['context_text'] for x in batch]
        abstracts = [x['abstract_text'] for x in batch]
        metas = [x['meta_text'] for x in batch]
        labels = torch.tensor([x['labels'] for x in batch], dtype=torch.float32)

        image_inputs = self.image_processor(images=images, return_tensors='pt')
        cap_inputs = self._encode_text(captions)
        ctx_inputs = self._encode_text(contexts)
        abs_inputs = self._encode_text(abstracts)
        meta_inputs = self._encode_text(metas)

        return {
            'pixel_values': image_inputs['pixel_values'],
            'caption_input_ids': cap_inputs['input_ids'],
            'caption_attention_mask': cap_inputs['attention_mask'],
            'context_input_ids': ctx_inputs['input_ids'],
            'context_attention_mask': ctx_inputs['attention_mask'],
            'abstract_input_ids': abs_inputs['input_ids'],
            'abstract_attention_mask': abs_inputs['attention_mask'],
            'meta_input_ids': meta_inputs['input_ids'],
            'meta_attention_mask': meta_inputs['attention_mask'],
            'labels': labels,
            'raw': [x['raw'] for x in batch],
        }
