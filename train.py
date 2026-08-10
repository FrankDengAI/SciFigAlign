import argparse
import json
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DIMENSIONS, FigureCollator, FigureQualityDataset
from src.models.model import ExplainableCrossAttentionCubeMLPRegressor, build_processors
from src.utils.device import resolve_device, setup_torch_backends
from src.utils.io import ensure_dir, write_json
from src.utils.metrics import summarize_metrics
from src.utils.export_data import flatten_metrics, write_csv_file, write_json_file
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Event emitter (consumed by backend run_manager.py)
# ---------------------------------------------------------------------------

def emit(event: dict) -> None:
    print(f"EVENT_JSON:{json.dumps(event, ensure_ascii=False)}", flush=True)


# ---------------------------------------------------------------------------
# Pairwise Ranking Loss
# ---------------------------------------------------------------------------

def pairwise_ranking_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
    min_diff: float = 0.5,
) -> torch.Tensor:
    """
    Batch-内构造样本对，学习相对排序关系。

    对 batch 中每对样本 (i, j)：
      - 真实差 diff = labels[i] - labels[j]
      - 只考虑 |diff| > min_diff 的有意义的对
      - 希望 sign(diff) * (preds[i] - preds[j]) > margin
      - 否则施加 hinge penalty: max(0, margin - sign(diff) * pred_diff)

    Args:
        preds:   [B, 4]  预测分数
        labels:  [B, 4]  真实标签
        margin:  排序间隔 (默认 1.0)
        min_diff: 真实分差至少要超过此值才构成有效对 (默认 0.5)

    Returns:
        loss: 标量
    """
    B = preds.size(0)
    if B < 2:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)

    # [B, B, 4]
    true_diff = labels.unsqueeze(0) - labels.unsqueeze(1)   # [B, B, 4]
    pred_diff = preds.unsqueeze(0) - preds.unsqueeze(1)     # [B, B, 4]

    # 只取上三角（避免重复对）
    upper_mask = torch.triu(torch.ones(B, B, device=preds.device, dtype=torch.bool), diagonal=1)
    upper_mask = upper_mask.unsqueeze(-1).expand_as(true_diff)  # [B, B, 4]

    # 只处理分差足够大的对
    valid = (torch.abs(true_diff) > min_diff) & upper_mask  # [B, B, 4]

    if not valid.any():
        return torch.tensor(0.0, device=preds.device, requires_grad=True)

    sign = torch.sign(true_diff)                            # [B, B, 4]
    hinge = torch.relu(margin - sign * pred_diff)           # [B, B, 4]

    return hinge[valid].mean()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train Figure Quality Regressor')

    # data
    p.add_argument('--train_jsonl', required=True)
    p.add_argument('--val_jsonl', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--verify_images', action='store_true')
    p.add_argument('--include_weak_labels', action='store_true')
    p.add_argument('--no_denoise_context', action='store_true',
                   help='Disable citing-context denoising (ablation: full_raw)')

    # model
    p.add_argument('--text_model_name', default='allenai/scibert_scivocab_uncased')
    p.add_argument('--image_model_name', default='openai/clip-vit-base-patch32')
    p.add_argument('--proj_dim', type=int, default=256)
    p.add_argument('--cube_tokens', type=int, default=4)
    p.add_argument('--cube_depth', type=int, default=3)
    p.add_argument('--mixing_expansion', type=float, default=2.0)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--cross_attention_heads', type=int, default=4)
    p.add_argument('--cross_attention_layers', type=int, default=1)
    p.add_argument('--freeze_text_encoder', action='store_true')
    p.add_argument('--freeze_image_encoder', action='store_true')
    p.add_argument('--use_bidirectional_cross_attention', action='store_true')

    # training
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--max_text_length', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_every_epoch', action='store_true')

    # pairwise ranking loss
    p.add_argument('--use_ranking_loss', action='store_true',
                   help='Enable pairwise ranking loss alongside regression loss')
    p.add_argument('--ranking_weight', type=float, default=0.2,
                   help='Weight of ranking loss in total loss (default: 0.2)')
    p.add_argument('--ranking_margin', type=float, default=1.0,
                   help='Margin for pairwise ranking hinge loss')
    p.add_argument('--ranking_min_diff', type=float, default=0.5,
                   help='Minimum label difference to form a valid pair')

    # logging
    p.add_argument('--log_every', type=int, default=10,
                   help='Print log every N steps')
    p.add_argument('--emit_every', type=int, default=1,
                   help='Emit EVENT_JSON every N steps')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    keys = [
        'pixel_values',
        'caption_input_ids', 'caption_attention_mask',
        'context_input_ids', 'context_attention_mask',
        'abstract_input_ids', 'abstract_attention_mask',
        'meta_input_ids', 'meta_attention_mask',
        'labels',
    ]
    return {k: batch[k].to(device, non_blocking=True) for k in keys}


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, best_val: float, args: argparse.Namespace) -> None:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_val': best_val,
        'args': vars(args),
    }, path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, Dict]:
    model.eval()
    losses: List[float] = []
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval', leave=False, file=sys.stderr):
            b = move_batch(batch, device)
            out = model(**{k: v for k, v in b.items() if k != 'labels'})
            preds = out['scores']
            loss = criterion(preds, b['labels'])
            losses.append(loss.item())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(b['labels'].cpu().numpy())

    preds_np = np.concatenate(all_preds, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    metrics = summarize_metrics(preds_np, labels_np)
    return float(np.mean(losses)) if losses else 0.0, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    # Device
    device = resolve_device('auto')
    setup_torch_backends(device)

    emit({'type': 'status', 'stage': 'loading', 'message': f'device={device}'})

    # Processors & collator
    tokenizer, image_processor = build_processors(args.text_model_name, args.image_model_name)
    collator = FigureCollator(image_processor, tokenizer, max_text_length=args.max_text_length)

    # Datasets
    train_ds = FigureQualityDataset(
        args.train_jsonl,
        verify_images=args.verify_images,
        include_weak_labels=args.include_weak_labels,
        denoise_context=not args.no_denoise_context,
    )
    val_ds = FigureQualityDataset(
        args.val_jsonl,
        verify_images=args.verify_images,
        include_weak_labels=args.include_weak_labels,
        denoise_context=not args.no_denoise_context,
    )
    emit({'type': 'dataset', 'train_size': len(train_ds), 'val_size': len(val_ds)})
    print(f'[dataset] train={len(train_ds)}, val={len(val_ds)}', flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collator,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collator,
    )

    # Model
    model = ExplainableCrossAttentionCubeMLPRegressor(
        text_model_name=args.text_model_name,
        image_model_name=args.image_model_name,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        cube_tokens=args.cube_tokens,
        cube_depth=args.cube_depth,
        mixing_expansion=args.mixing_expansion,
        freeze_text_encoder=args.freeze_text_encoder,
        freeze_image_encoder=args.freeze_image_encoder,
        cross_attention_heads=args.cross_attention_heads,
        cross_attention_layers=args.cross_attention_layers,
        use_bidirectional_cross_attention=args.use_bidirectional_cross_attention,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[model] total_params={total_params:,}, trainable={trainable_params:,}', flush=True)

    emit({'type': 'config', 'log_every': args.log_every, 'emit_every': args.emit_every})

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    reg_criterion = nn.SmoothL1Loss(beta=1.0)

    # Save args
    # write_json(vars(args), f'{args.output_dir}/train_args.json')
    write_json(f'{args.output_dir}/train_args.json', vars(args))

    history: List[Dict] = []
    batch_export_rows: List[Dict] = []
    best_val_loss = float('inf')
    total_steps = len(train_loader)

    emit({'type': 'status', 'stage': 'running', 'message': 'training started'})

    for epoch in range(1, args.epochs + 1):
        model.train()
        emit({'type': 'epoch_start', 'epoch': epoch, 'total_epochs': args.epochs,
              'total_steps': total_steps})

        train_reg_losses: List[float] = []
        train_rank_losses: List[float] = []
        train_total_losses: List[float] = []

        progress = tqdm(train_loader, desc=f'[epoch {epoch}/{args.epochs}]', file=sys.stderr)
        for step, batch in enumerate(progress, start=1):
            b = move_batch(batch, device)
            out = model(**{k: v for k, v in b.items() if k != 'labels'})
            preds = out['scores']
            labels = b['labels']

            # Regression loss (SmoothL1)
            reg_loss = reg_criterion(preds, labels)

            # Pairwise ranking loss
            if args.use_ranking_loss:
                rank_loss = pairwise_ranking_loss(
                    preds, labels,
                    margin=args.ranking_margin,
                    min_diff=args.ranking_min_diff,
                )
                total_loss = reg_loss + args.ranking_weight * rank_loss
            else:
                rank_loss = torch.tensor(0.0, device=device)
                total_loss = reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            reg_l = reg_loss.item()
            rank_l = rank_loss.item()
            total_l = total_loss.item()
            train_reg_losses.append(reg_l)
            train_rank_losses.append(rank_l)
            train_total_losses.append(total_l)

            attn_peak = out.get('cross_attention_peak')
            attn_entropy = out.get('cross_attention_entropy')
            attn_peak_val = float(attn_peak.mean().item()) if attn_peak is not None else None
            attn_entropy_val = float(attn_entropy.mean().item()) if attn_entropy is not None else None

            progress.set_postfix(
                loss=f'{total_l:.4f}',
                reg=f'{reg_l:.4f}',
                rank=f'{rank_l:.4f}',
            )

            # Print line (captured by run_manager EPOCH_RE / LOSS_RE)
            if step % args.log_every == 0 or step == total_steps:
                print(
                    f'[epoch {epoch}/{args.epochs}] step {step}/{total_steps} '
                    f'loss={total_l:.4f} reg={reg_l:.4f} rank={rank_l:.4f}',
                    flush=True,
                )

            # Emit structured event (captured by run_manager EVENT_JSON)
            batch_export_rows.append({
                'global_step': (epoch - 1) * total_steps + step,
                'epoch': epoch,
                'step': step,
                'total_steps': total_steps,
                'loss': total_l,
                'reg_loss': reg_l,
                'rank_loss': rank_l,
                'attn_peak': attn_peak_val,
                'attn_entropy': attn_entropy_val,
            })

            if step % args.emit_every == 0 or step == total_steps:
                emit({
                    'type': 'batch',
                    'epoch': epoch,
                    'total_epochs': args.epochs,
                    'step': step,
                    'total_steps': total_steps,
                    'loss': total_l,
                    'reg_loss': reg_l,
                    'rank_loss': rank_l,
                    'attn_peak': attn_peak_val,
                    'attn_entropy': attn_entropy_val,
                })

        # --- Epoch end ---
        train_loss_avg = float(np.mean(train_total_losses)) if train_total_losses else 0.0
        val_loss, val_metrics = evaluate(model, val_loader, device, reg_criterion)

        avg_spearman = val_metrics.get('macro_avg', {}).get('spearman', 0.0)
        metric_snapshot = {
            dim: {
                'spearman': val_metrics[dim]['spearman'],
                'mae': val_metrics[dim]['mae'],
            }
            for dim in DIMENSIONS
        }

        record = {
            'epoch': epoch,
            'train_loss': train_loss_avg,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'ranking_loss_avg': float(np.mean(train_rank_losses)) if train_rank_losses else 0.0,
        }
        history.append(record)
        # Raw data used by all training-monitor visualisations.
        write_json(f'{args.output_dir}/history.json', {'history': history})
        write_json_file(f'{args.output_dir}/visualization_data/training_batch_history.json', batch_export_rows)
        write_csv_file(f'{args.output_dir}/visualization_data/training_batch_history.csv', batch_export_rows)

        epoch_rows = []
        metric_rows = []
        for item in history:
            epoch_rows.append({
                'epoch': item['epoch'],
                'train_loss': item['train_loss'],
                'val_loss': item['val_loss'],
                'ranking_loss_avg': item.get('ranking_loss_avg', 0.0),
            })
            for dimension, metrics in item.get('val_metrics', {}).items():
                if isinstance(metrics, dict):
                    metric_rows.append({'epoch': item['epoch'], 'dimension': dimension, **metrics})
        write_json_file(f'{args.output_dir}/visualization_data/training_epoch_history.json', history)
        write_csv_file(f'{args.output_dir}/visualization_data/training_epoch_loss.csv', epoch_rows)
        write_csv_file(f'{args.output_dir}/visualization_data/training_epoch_metrics.csv', metric_rows)

        print(
            f'[epoch {epoch}/{args.epochs}] train_loss={train_loss_avg:.4f} '
            f'val_loss={val_loss:.4f} spearman={avg_spearman:.4f}',
            flush=True,
        )

        attn_peak_epoch = float(np.mean([
            x for x in [
                b.get('attn_peak') for b in (
                    [{'attn_peak': attn_peak_val}] if attn_peak_val is not None else []
                )
            ] if x is not None
        ])) if attn_peak_val is not None else None

        emit({
            'type': 'epoch_end',
            'epoch': epoch,
            'total_epochs': args.epochs,
            'train_loss': train_loss_avg,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'metric_snapshot': metric_snapshot,
            'attn_peak': attn_peak_epoch,
            'attn_entropy': None,
        })

        # Checkpoint
        if args.save_every_epoch:
            save_checkpoint(
                f'{args.output_dir}/epoch_{epoch}.pt',
                model, optimizer, epoch, best_val_loss, args,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(f'{args.output_dir}/best.pt', model, optimizer, epoch, best_val_loss, args)
            print(f'[checkpoint] epoch={epoch} best_val={best_val_loss:.4f}', flush=True)
            emit({'type': 'best_checkpoint', 'epoch': epoch, 'best_val': best_val_loss})

    # --- Training done ---
    save_checkpoint(f'{args.output_dir}/last.pt', model, optimizer, args.epochs, best_val_loss, args)
    emit({'type': 'finished', 'best_val': best_val_loss})
    print(f'[done] best_val={best_val_loss:.4f}', flush=True)


if __name__ == '__main__':
    main()
