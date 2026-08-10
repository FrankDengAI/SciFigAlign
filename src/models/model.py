from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel

from src.dataset import DIMENSIONS


MODALITY_NAMES = ['image', 'caption', 'context', 'abstract', 'meta']
TEXT_MODALITIES = ['caption', 'context', 'abstract', 'meta']


def build_processors(text_model_name: str, image_model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    image_processor = CLIPImageProcessor.from_pretrained(image_model_name)
    return tokenizer, image_processor


class AxisMLP(nn.Module):
    def __init__(self, axis_size: int, expansion: float = 2.0, dropout: float = 0.1):
        super().__init__()
        hidden = max(int(axis_size * expansion), axis_size)
        self.net = nn.Sequential(
            nn.Linear(axis_size, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, axis_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = max(int(dim * expansion), dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CubeMLPBlock(nn.Module):
    def __init__(self, seq_len: int, num_modalities: int, dim: int, mixing_expansion: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.seq_mlp = AxisMLP(seq_len, expansion=mixing_expansion, dropout=dropout)
        self.mod_mlp = AxisMLP(num_modalities, expansion=mixing_expansion, dropout=dropout)
        self.chn_mlp = FeedForward(dim, expansion=max(2.0, mixing_expansion), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        seq_view = y.permute(0, 2, 3, 1)
        x = x + self.seq_mlp(seq_view).permute(0, 3, 1, 2)

        y = self.norm(x)
        mod_view = y.permute(0, 1, 3, 2)
        x = x + self.mod_mlp(mod_view).permute(0, 1, 3, 2)

        y = self.norm(x)
        x = x + self.chn_mlp(y)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, expansion=4.0, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_tokens: torch.Tensor,
        kv_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(
            query=query_tokens,
            key=kv_tokens,
            value=kv_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm1(query_tokens + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, attn_weights


class ExplainableCrossAttentionCubeMLPRegressor(nn.Module):
    def __init__(
        self,
        text_model_name: str = 'allenai/scibert_scivocab_uncased',
        image_model_name: str = 'openai/clip-vit-base-patch32',
        proj_dim: int = 256,
        dropout: float = 0.1,
        cube_tokens: int = 4,
        cube_depth: int = 3,
        mixing_expansion: float = 2.0,
        freeze_text_encoder: bool = False,
        freeze_image_encoder: bool = False,
        cross_attention_heads: int = 4,
        cross_attention_layers: int = 1,
        use_bidirectional_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.image_encoder = CLIPVisionModel.from_pretrained(image_model_name)
        self.cube_tokens = cube_tokens
        self.modality_names: List[str] = list(MODALITY_NAMES)
        self.use_bidirectional_cross_attention = use_bidirectional_cross_attention

        text_hidden = self.text_encoder.config.hidden_size
        image_hidden = self.image_encoder.config.hidden_size

        self.caption_proj = nn.Linear(text_hidden, proj_dim)
        self.context_proj = nn.Linear(text_hidden, proj_dim)
        self.abstract_proj = nn.Linear(text_hidden, proj_dim)
        self.meta_proj = nn.Linear(text_hidden, proj_dim)
        self.image_proj = nn.Linear(image_hidden, proj_dim)

        self.cross_attn_text_to_image = nn.ModuleList([
            CrossAttentionBlock(proj_dim, num_heads=cross_attention_heads, dropout=dropout)
            for _ in range(cross_attention_layers)
        ])
        self.cross_attn_image_to_text = nn.ModuleList([
            CrossAttentionBlock(proj_dim, num_heads=cross_attention_heads, dropout=dropout)
            for _ in range(cross_attention_layers)
        ])

        self.stream_type = nn.Parameter(torch.randn(len(self.modality_names), proj_dim) * 0.02)
        self.blocks = nn.ModuleList([
            CubeMLPBlock(
                seq_len=cube_tokens,
                num_modalities=len(self.modality_names),
                dim=proj_dim,
                mixing_expansion=mixing_expansion,
                dropout=dropout,
            )
            for _ in range(cube_depth)
        ])
        self.final_norm = nn.LayerNorm(proj_dim)

        self.global_modality_gate = nn.Sequential(
            nn.Linear(proj_dim, max(proj_dim // 2, 8)),
            nn.GELU(),
            nn.Linear(max(proj_dim // 2, 8), 1),
        )
        self.dim_queries = nn.Parameter(torch.randn(len(DIMENSIONS), proj_dim) * 0.02)
        self.score_heads = nn.ModuleDict({
            dim: nn.Sequential(
                nn.Linear(proj_dim * 2, proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, max(proj_dim // 2, 8)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(proj_dim // 2, 8), 1),
            )
            for dim in DIMENSIONS
        })

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    def _adaptive_token_pool(self, states: torch.Tensor, target_tokens: int) -> torch.Tensor:
        if states.size(1) == target_tokens:
            return states
        x = states.transpose(1, 2)
        x = F.adaptive_avg_pool1d(x, target_tokens)
        return x.transpose(1, 2)

    def _adaptive_masked_token_pool(self, states: torch.Tensor, attention_mask: torch.Tensor, target_tokens: int) -> torch.Tensor:
        weights = attention_mask.float().unsqueeze(-1)
        masked = states * weights
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        global_mean = masked.sum(dim=1, keepdim=True) / denom
        pooled = self._adaptive_token_pool(masked, target_tokens)
        pooled_mask = self._adaptive_token_pool(weights, target_tokens)
        pooled = pooled / pooled_mask.clamp(min=1e-6)
        valid = (pooled_mask > 0).float()
        pooled = pooled * valid + global_mean * (1.0 - valid)
        return pooled

    def _encode_text_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, proj: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_states = outputs.last_hidden_state
        pooled_tokens = self._adaptive_masked_token_pool(token_states, attention_mask, self.cube_tokens)
        return proj(pooled_tokens), proj(token_states)

    def _mean_heads(self, attn_weights: torch.Tensor) -> torch.Tensor:
        # [B, heads, q, k] -> [B, q, k]
        return attn_weights.mean(dim=1)

    def _run_cross_attention(self, text_tokens: torch.Tensor, image_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        text_to_image_maps = []
        image_to_text_maps = []
        current_text = text_tokens
        current_image = image_tokens
        for layer_idx, block in enumerate(self.cross_attn_text_to_image):
            current_text, attn_weights = block(current_text, current_image)
            text_to_image_maps.append(self._mean_heads(attn_weights))
            if self.use_bidirectional_cross_attention and layer_idx < len(self.cross_attn_image_to_text):
                current_image, image_attn = self.cross_attn_image_to_text[layer_idx](current_image, current_text)
                image_to_text_maps.append(self._mean_heads(image_attn))
        aux = {
            'text_to_image_layers': torch.stack(text_to_image_maps, dim=1) if text_to_image_maps else None,
            'image_to_text_layers': torch.stack(image_to_text_maps, dim=1) if image_to_text_maps else None,
        }
        return current_text, current_image, aux

    def _build_cube(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_outputs = self.image_encoder(pixel_values=batch['pixel_values'])
        image_patch_tokens_raw = image_outputs.last_hidden_state[:, 1:, :]
        image_patch_tokens = self.image_proj(image_patch_tokens_raw)
        image_tokens = self._adaptive_token_pool(image_patch_tokens, self.cube_tokens)

        text_specs = [
            ('caption', batch['caption_input_ids'], batch['caption_attention_mask'], self.caption_proj),
            ('context', batch['context_input_ids'], batch['context_attention_mask'], self.context_proj),
            ('abstract', batch['abstract_input_ids'], batch['abstract_attention_mask'], self.abstract_proj),
            ('meta', batch['meta_input_ids'], batch['meta_attention_mask'], self.meta_proj),
        ]

        streams = [image_tokens]
        cross_attention_summary: Dict[str, Dict[str, torch.Tensor]] = {}
        attended_image_views = []
        raw_text_tokens = {}
        for name, input_ids, attention_mask, proj in text_specs:
            pooled_tokens, raw_tokens = self._encode_text_tokens(input_ids, attention_mask, proj)
            raw_text_tokens[f'{name}_raw_tokens'] = raw_tokens
            attended_text, attended_image, cross_aux = self._run_cross_attention(pooled_tokens, image_patch_tokens)
            streams.append(attended_text)
            attended_image_views.append(self._adaptive_token_pool(attended_image, self.cube_tokens))

            t2i = cross_aux['text_to_image_layers']
            t2i_mean = t2i.mean(dim=1) if t2i is not None else None  # [B, q, patch]
            summary = {
                'patch_focus': t2i_mean.mean(dim=1) if t2i_mean is not None else None,
                'token_focus': t2i_mean.mean(dim=2) if t2i_mean is not None else None,
            }
            if cross_aux['image_to_text_layers'] is not None:
                i2t_mean = cross_aux['image_to_text_layers'].mean(dim=1)
                summary['reverse_patch_focus'] = i2t_mean.mean(dim=2)
                summary['reverse_token_focus'] = i2t_mean.mean(dim=1)
            cross_attention_summary[name] = summary

        if attended_image_views:
            image_tokens = image_tokens + torch.stack(attended_image_views, dim=0).mean(dim=0)
            streams[0] = image_tokens

        stream_tensor = torch.stack(streams, dim=2)
        stream_tensor = stream_tensor + self.stream_type.view(1, 1, len(self.modality_names), -1)

        aux = {
            **raw_text_tokens,
            'image_patch_tokens': image_patch_tokens,
            'image_patch_count': image_patch_tokens.shape[1],
            'cross_attention_summary': cross_attention_summary,
        }
        return stream_tensor, aux

    def _build_explanations(self, cube: torch.Tensor, aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor | Dict]:
        token_strength = cube.norm(dim=-1)
        token_importance = torch.softmax(token_strength.permute(0, 2, 1), dim=-1)

        modality_context = cube.mean(dim=1)
        global_modality_logits = self.global_modality_gate(modality_context).squeeze(-1)
        modality_importance = torch.softmax(global_modality_logits, dim=-1)

        dim_modality_weights = []
        dim_token_weights = []
        weighted_features = []
        for idx in range(len(DIMENSIONS)):
            query = self.dim_queries[idx].view(1, 1, -1)
            logits = (modality_context * query).sum(dim=-1)
            weights = torch.softmax(logits, dim=-1)
            dim_modality_weights.append(weights)

            token_logits = (cube * query.view(1, 1, 1, -1)).sum(dim=-1).permute(0, 2, 1)
            token_w = torch.softmax(token_logits, dim=-1)
            dim_token_weights.append(token_w)
            weighted_features.append((modality_context * weights.unsqueeze(-1)).sum(dim=1))

        cross_attention = aux.get('cross_attention_summary', {})
        cross_modality_focus = []
        for name in TEXT_MODALITIES:
            patch_focus = cross_attention.get(name, {}).get('patch_focus')
            if patch_focus is None:
                cross_modality_focus.append(torch.zeros(cube.size(0), device=cube.device))
            else:
                cross_modality_focus.append(patch_focus.max(dim=-1).values)
        cross_modality_focus = torch.stack(cross_modality_focus, dim=-1) if cross_modality_focus else None
        if cross_modality_focus is not None:
            cross_modality_focus = torch.softmax(cross_modality_focus, dim=-1)

        return {
            'modality_importance': modality_importance,
            'token_importance': token_importance,
            'dimension_modality_importance': torch.stack(dim_modality_weights, dim=1),
            'dimension_token_importance': torch.stack(dim_token_weights, dim=1),
            'dimension_features': torch.stack(weighted_features, dim=1),
            'cross_attention_summary': cross_attention,
            'cross_modality_focus': cross_modality_focus,
        }

    def _cross_attention_stats(self, explanations: Dict[str, torch.Tensor | Dict]) -> Dict[str, torch.Tensor]:
        peaks = []
        entropies = []
        cross_attention = explanations.get('cross_attention_summary', {})
        for name in TEXT_MODALITIES:
            patch_focus = cross_attention.get(name, {}).get('patch_focus')
            if patch_focus is None:
                continue
            probs = patch_focus / patch_focus.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            peaks.append(probs.max(dim=-1).values)
            entropies.append((-(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)))
        if not peaks:
            zero = torch.zeros(1, device=next(self.parameters()).device)
            return {'attn_peak': zero, 'attn_entropy': zero}
        return {
            'attn_peak': torch.stack(peaks, dim=0).mean(dim=0),
            'attn_entropy': torch.stack(entropies, dim=0).mean(dim=0),
        }

    def forward(self, **batch) -> Dict[str, torch.Tensor | Dict | List[str]]:
        cube, aux = self._build_cube(batch)
        for block in self.blocks:
            cube = block(cube)
        cube = self.final_norm(cube)

        explanations = self._build_explanations(cube, aux)
        global_feature = cube.mean(dim=(1, 2))
        dim_features = explanations['dimension_features']

        scores = []
        for idx, dim in enumerate(DIMENSIONS):
            feat = torch.cat([global_feature, dim_features[:, idx, :]], dim=-1)
            scores.append(torch.sigmoid(self.score_heads[dim](feat)) * 5.0)
        scores = torch.cat(scores, dim=-1)
        stats = self._cross_attention_stats(explanations)

        return {
            'scores': scores,
            'fused': global_feature,
            'cube': cube,
            'modality_importance': explanations['modality_importance'],
            'token_importance': explanations['token_importance'],
            'dimension_modality_importance': explanations['dimension_modality_importance'],
            'dimension_token_importance': explanations['dimension_token_importance'],
            'cross_modality_focus': explanations['cross_modality_focus'],
            'cross_attention_summary': explanations['cross_attention_summary'],
            'cross_attention_peak': stats['attn_peak'],
            'cross_attention_entropy': stats['attn_entropy'],
            'modality_names': self.modality_names,
            'aux': aux,
        }


class MultiStreamFigureRegressor(ExplainableCrossAttentionCubeMLPRegressor):
    pass
