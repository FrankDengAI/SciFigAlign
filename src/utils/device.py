from __future__ import annotations

from typing import Any, Dict

import torch


def resolve_device(device_arg: str = 'auto') -> torch.device:
    choice = (device_arg or 'auto').lower()
    if choice == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    if choice == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but torch.cuda.is_available() is False.')
        return torch.device('cuda')
    if choice == 'mps':
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            raise RuntimeError('MPS was requested but is not available in this environment.')
        return torch.device('mps')
    if choice == 'cpu':
        return torch.device('cpu')
    raise ValueError(f'Unsupported device option: {device_arg}')


class NullScaler:
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def unscale_(self, optimizer) -> None:
        return None


def create_grad_scaler(device: torch.device, enable_amp: bool):
    if device.type == 'cuda' and enable_amp:
        return torch.amp.GradScaler('cuda')
    return NullScaler()


def get_autocast_context(device: torch.device, enable_amp: bool):
    if device.type == 'cuda' and enable_amp:
        return torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    if device.type == 'mps' and enable_amp:
        return torch.amp.autocast(device_type='cpu', enabled=False)
    return torch.amp.autocast(device_type='cpu', enabled=False)


def can_use_non_blocking(device: torch.device) -> bool:
    return device.type == 'cuda'


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    non_blocking = can_use_non_blocking(device)
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v
    return out


def setup_torch_backends(device: torch.device, allow_tf32: bool = True) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.allow_tf32 = allow_tf32
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_count'] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        info['total_vram_gb'] = round(props.total_memory / (1024 ** 3), 2)
    else:
        info['gpu_name'] = ''
        info['gpu_count'] = 0
        info['total_vram_gb'] = 0.0
    info['allow_tf32'] = bool(allow_tf32 and device.type == 'cuda')
    return info
