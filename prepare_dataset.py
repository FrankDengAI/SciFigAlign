import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

from src.path_config import resolve_figure_path
from src.utils.io import read_jsonl, write_jsonl, ensure_dir

DIMENSIONS = ['clarity', 'relevance', 'informativeness', 'structure']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--figures_jsonl', required=True)
    p.add_argument('--papers_jsonl', required=True)
    p.add_argument('--labels_jsonl', default=None)
    p.add_argument('--output_jsonl', required=True)
    p.add_argument('--image_path_prefix_to', default=None)
    p.add_argument('--allow_weak_labels', action='store_true')
    return p.parse_args()


def normalize_path(path: str, prefix_to: Optional[str]) -> str:
    if not path:
        return path

    path = path.replace("\\", "/")

    old_prefixes = [
        "/Volumes/JHAO/data_v1",
        "/Volumes/JHAO/data",
    ]

    if prefix_to:
        prefix_to = prefix_to.replace("\\", "/").rstrip("/")
        for old in old_prefixes:
            if path.startswith(old):
                return prefix_to + path[len(old):]

    try:
        return resolve_figure_path(path)
    except FileNotFoundError:
        return path


def weak_scores_from_paper(decision: str, avg_rating: Optional[float]) -> Dict[str, Dict[str, float]]:
    decision = (decision or '').lower()
    base = float(avg_rating) if avg_rating is not None else 5.0
    if 'reject' in decision:
        offset = -1.0
    elif 'accept' in decision:
        offset = 0.5
    else:
        offset = 0.0
    score = max(1.0, min(5.0, round(base / 2.0 + offset)))
    return {d: {'score': float(score), 'rationale': 'weak label from paper metadata'} for d in DIMENSIONS}


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.output_jsonl)
    if out_dir:
        ensure_dir(out_dir)

    figures = read_jsonl(args.figures_jsonl)
    papers = {x['paper_id']: x for x in read_jsonl(args.papers_jsonl)}
    label_map = {}
    if args.labels_jsonl:
        for x in read_jsonl(args.labels_jsonl):
            label_map[(x['paper_id'], x['figure_id'])] = x

    merged: List[Dict] = []
    for fig in figures:
        paper = papers.get(fig['paper_id'], {})
        label_row = label_map.get((fig['paper_id'], fig['figure_id']))
        row = {
            'paper_id': fig.get('paper_id', ''),
            'paper_title': paper.get('title', ''),
            'year': paper.get('year'),
            'venue': paper.get('venue', ''),
            'decision': paper.get('decision', ''),
            'avg_rating': paper.get('avg_rating', None),
            'abstract': paper.get('abstract', ''),
            'figure_id': fig.get('figure_id', ''),
            'figure_image_path': normalize_path(fig.get('figure_image_path', ''), args.image_path_prefix_to),
            'figure_type': fig.get('figure_type', ''),
            'caption_text': fig.get('caption_text', ''),
            'citing_paragraphs': fig.get('citing_paragraphs', []),
            'section_title': fig.get('section_title', ''),
            'nearby_context': fig.get('nearby_context', []),
        }
        if label_row and label_row.get('scores'):
            row['scores'] = label_row['scores']
            row['label_source'] = label_row.get('annotator_id', 'human')
        elif args.allow_weak_labels:
            row['scores'] = weak_scores_from_paper(paper.get('decision', ''), paper.get('avg_rating'))
            row['label_source'] = 'weak'
        else:
            continue
        merged.append(row)

    write_jsonl(args.output_jsonl, merged)
    print(f'Saved {len(merged)} merged rows to {args.output_jsonl}')


if __name__ == '__main__':
    main()
