"""Multiplicity-aware self-supervised losses in the lifted domain."""
from __future__ import annotations
import torch

def _expand_physical_tensor(value: torch.Tensor, *, batch_size: int, num_coils: int, height: int, width: int, name: str) -> torch.Tensor:
    """Expand a shared or coil-resolved physical array to ``[B,C,H,W]``."""
    if value.ndim == 2 and tuple(value.shape) == (height, width):
        return value[None, None].expand(batch_size, num_coils, -1, -1)
    if value.ndim == 3 and tuple(value.shape) == (batch_size, height, width):
        return value[:, None].expand(-1, num_coils, -1, -1)
    if value.ndim == 4 and tuple(value.shape) == (batch_size, num_coils, height, width):
        return value
    raise ValueError(f'{name} must have shape {(height, width)}, {(batch_size, height, width)}, or {(batch_size, num_coils, height, width)}, got {tuple(value.shape)}')

def physical_ipw_heldout_sum(prediction: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor, inclusion_probability: torch.Tensor, point_weight: torch.Tensor | None=None) -> torch.Tensor:
    """Return the unnormalized physical-coordinate IPW squared-error sum.

    ``point_weight`` is an optional non-negative physical-coordinate weight.
    It is intended for weights computed from the input view(s) only (for
    example, an acquired-only uncertainty estimate); the target values remain
    used only in the squared error.  The supplied inverse probability is a
    pre-draw design weight.  A fixed-predictor condition is required for an
    unbiased Horvitz--Thompson interpretation; this function does not
    establish that condition for an online shared-parameter trainer.  The
    default path is unchanged.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError('prediction and target must share shape [B,C,H,W]')
    if not torch.is_complex(prediction) or not torch.is_complex(target):
        raise ValueError('prediction and target must be complex')
    (batch_size, num_coils, height, width) = target.shape
    support = _expand_physical_tensor(target_mask, batch_size=batch_size, num_coils=num_coils, height=height, width=width, name='target_mask').bool()
    probability = _expand_physical_tensor(inclusion_probability, batch_size=batch_size, num_coils=num_coils, height=height, width=width, name='inclusion_probability').to(dtype=prediction.real.dtype, device=prediction.device)
    if point_weight is None:
        weight = torch.ones_like(probability)
    else:
        weight = _expand_physical_tensor(point_weight, batch_size=batch_size, num_coils=num_coils, height=height, width=width, name='point_weight').to(dtype=prediction.real.dtype, device=prediction.device)
        if torch.any(~torch.isfinite(weight)) or torch.any(weight < 0.0):
            raise ValueError('point_weight must be finite and non-negative')
    if torch.any(~torch.isfinite(probability)) or torch.any(support & (probability <= 0.0)):
        raise ValueError('selected target coordinates require finite positive probabilities')
    squared_error = torch.abs(prediction - target).square()
    safe_probability = torch.where(support, probability, torch.ones_like(probability))
    return (squared_error * support * weight / safe_probability).sum()

