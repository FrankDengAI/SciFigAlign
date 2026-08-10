import argparse
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DIMENSIONS, FigureCollator, FigureQualityDataset, InferenceFigureDataset
from src.models.model import MultiStreamFigureRegressor, build_processors
from src.utils.io import write_jsonl
from src.utils.device import move_to_device, resolve_device, setup_torch_backends


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input_jsonl', default=None)
    p.add_argument('--output_jsonl', default=None)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--verify_images', action='store_true')
    p.add_argument('--include_weak_labels', action='store_true')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu', 'mps'])
    return p.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    train_args = ckpt['args']
    tokenizer, image_processor = build_processors(train_args['text_model_name'], train_args['image_model_name'])
    model = MultiStreamFigureRegressor(
        text_model_name=train_args['text_model_name'],
        image_model_name=train_args['image_model_name'],
        proj_dim=train_args['proj_dim'],
        dropout=train_args['dropout'],
        cube_tokens=train_args.get('cube_tokens', 4),
        cube_depth=train_args.get('cube_depth', 3),
        mixing_expansion=train_args.get('mixing_expansion', 2.0),
        freeze_text_encoder=train_args.get('freeze_text_encoder', False),
        freeze_image_encoder=train_args.get('freeze_image_encoder', False),
        cross_attention_heads=train_args.get('cross_attention_heads', 4),
        cross_attention_layers=train_args.get('cross_attention_layers', 1),
        use_bidirectional_cross_attention=train_args.get('use_bidirectional_cross_attention', False),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    collator = FigureCollator(image_processor=image_processor, tokenizer=tokenizer, max_text_length=train_args['max_text_length'])
    return model, collator, train_args


def predict_rows(model, loader, device):
    rows: List[Dict] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Predicting'):
            tensor_batch = move_to_device(batch, device)
            outputs = model(**tensor_batch)
            preds = outputs['scores'].cpu().tolist()
            for i, pred in enumerate(preds):
                raw = batch['raw'][i]
                pred_scores = {dim: float(pred[j]) for j, dim in enumerate(DIMENSIONS)}
                gold_scores = {}
                for dim in DIMENSIONS:
                    value = raw.get('scores', {}).get(dim, 0)
                    if isinstance(value, dict):
                        value = value.get('score', 0)
                    gold_scores[dim] = float(value)
                modality_names = outputs.get('modality_names', ['image', 'caption', 'context', 'abstract', 'meta'])
                modality_importance = outputs.get('modality_importance')
                dim_modality_importance = outputs.get('dimension_modality_importance')
                explanation = {}
                if modality_importance is not None:
                    global_w = modality_importance[i].detach().cpu().tolist()
                    explanation['global_modality_importance'] = {name: float(global_w[idx]) for idx, name in enumerate(modality_names)}
                if dim_modality_importance is not None:
                    dim_w = dim_modality_importance[i].detach().cpu().tolist()
                    explanation['dimension_modality_importance'] = {
                        dim: {name: float(dim_w[d_idx][m_idx]) for m_idx, name in enumerate(modality_names)}
                        for d_idx, dim in enumerate(DIMENSIONS)
                    }
                    explanation['top_modality_per_dimension'] = {
                        dim: modality_names[max(range(len(modality_names)), key=lambda m_idx: dim_w[d_idx][m_idx])]
                        for d_idx, dim in enumerate(DIMENSIONS)
                    }
                cross_modality_focus = outputs.get('cross_modality_focus')
                cross_attention_summary = outputs.get('cross_attention_summary', {})
                if cross_modality_focus is not None:
                    cross_focus_vals = cross_modality_focus[i].detach().cpu().tolist()
                    explanation['cross_modality_focus'] = {name: float(cross_focus_vals[idx]) for idx, name in enumerate(['caption','context','abstract','meta'])}
                if cross_attention_summary:
                    explanation['cross_attention_patch_focus'] = {}
                    explanation['cross_attention_token_focus'] = {}
                    explanation['top_patch_index_by_modality'] = {}
                    for name, summary in cross_attention_summary.items():
                        patch_focus = summary.get('patch_focus')
                        token_focus = summary.get('token_focus')
                        if patch_focus is not None:
                            pvals = patch_focus[i].detach().cpu().tolist()
                            explanation['cross_attention_patch_focus'][name] = [float(x) for x in pvals]
                            explanation['top_patch_index_by_modality'][name] = int(max(range(len(pvals)), key=lambda idx: pvals[idx])) if pvals else -1
                        if token_focus is not None:
                            tvals = token_focus[i].detach().cpu().tolist()
                            explanation['cross_attention_token_focus'][name] = [float(x) for x in tvals]
                rows.append({
                    'paper_id': raw.get('paper_id', ''),
                    'paper_title': raw.get('paper_title', ''),
                    'figure_id': raw.get('figure_id', ''),
                    'figure_image_path': raw.get('figure_image_path', ''),
                    'figure_type': raw.get('figure_type', ''),
                    'caption_text': raw.get('caption_text', ''),
                    'pred_scores': pred_scores,
                    'gold_scores': gold_scores,
                    'explanation': explanation,
                    'label_source': raw.get('label_source', ''),
                })
    return rows


def score_single_image(model, collator, device, image, caption_text='', context_text='', abstract_text='', meta_text=''):
    ds = InferenceFigureDataset([{
        'image': image,
        'caption_text': caption_text,
        'context_text': context_text,
        'abstract_text': abstract_text,
        'meta_text': meta_text,
    }])
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator)
    rows = predict_rows(model, loader, device)
    row = rows[0]
    return {'scores': row['pred_scores'], 'explanation': row.get('explanation', {})}


def main():
    args = parse_args()
    device = resolve_device(args.device)
    backend_info = setup_torch_backends(device, allow_tf32=True)
    print(f"Using device: {device}")
    if backend_info.get('gpu_name'):
        print(f"GPU: {backend_info['gpu_name']} | VRAM: {backend_info['total_vram_gb']} GB")
    model, collator, _ = load_model(args.checkpoint, device)
    ds = FigureQualityDataset(args.input_jsonl, verify_images=args.verify_images, include_weak_labels=args.include_weak_labels)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)
    rows = predict_rows(model, loader, device)
    if args.output_jsonl:
        write_jsonl(args.output_jsonl, rows)
        print(f'Saved {len(rows)} predictions to {args.output_jsonl}')


if __name__ == '__main__':
    main()
