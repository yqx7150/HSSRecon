"""Explicit block-Hankel lifting utilities for MRI k-space."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def _validate_shape(shape: tuple[int, ...], kernel_size: tuple[int, int]) -> None:
    if len(shape) < 3:
        raise ValueError(f"Expected [...,C,H,W], got {shape}")
    kh, kw = kernel_size
    if kh > shape[-2] or kw > shape[-1]:
        raise ValueError(f"Kernel {kernel_size} does not fit spatial shape {shape[-2:]}")


def block_hankel_np(kspace: np.ndarray, kernel_size: tuple[int, int]) -> np.ndarray:
    """Lift [C,H,W] k-space into [num_patches,C*kh*kw]."""

    array = np.asarray(kspace)
    _validate_shape(array.shape, kernel_size)
    kh, kw = kernel_size
    windows = np.lib.stride_tricks.sliding_window_view(
        array, window_shape=(kh, kw), axis=(-2, -1)
    )
    return windows.transpose(1, 2, 0, 3, 4).reshape(-1, array.shape[0] * kh * kw)


def block_hankel_torch(
    kspace: "torch.Tensor", kernel_size: tuple[int, int]
) -> "torch.Tensor":
    """Lift ``[B,C,H,W]`` k-space to ``[B,num_rows,C*kh*kw]``."""

    if kspace.ndim != 4:
        raise ValueError(f"Expected kspace [B,C,H,W], got {tuple(kspace.shape)}")
    _validate_shape(tuple(kspace.shape), kernel_size)
    kh, kw = kernel_size
    windows = kspace.unfold(2, kh, 1).unfold(3, kw, 1)
    return windows.permute(0, 2, 3, 1, 4, 5).reshape(
        kspace.shape[0], -1, kspace.shape[1] * kh * kw
    )


def multiplicity_map_np(
    output_size: tuple[int, int], kernel_size: tuple[int, int]
) -> np.ndarray:
    """Count the Hankel copies associated with each physical coordinate."""

    height, width = output_size
    kh, kw = kernel_size
    _validate_shape((1, height, width), kernel_size)
    counts = np.zeros((height, width), dtype=np.float32)
    rows = height - kh + 1
    cols = width - kw + 1
    for iy in range(kh):
        for ix in range(kw):
            counts[iy : iy + rows, ix : ix + cols] += 1.0
    return counts


def multiplicity_map_torch(
    output_size: tuple[int, int],
    kernel_size: tuple[int, int],
    *,
    device: "torch.device",
) -> "torch.Tensor":
    import torch

    return torch.as_tensor(multiplicity_map_np(output_size, kernel_size), device=device)


def inverse_sqrt_multiplicity_map_torch(
    output_size: tuple[int, int],
    kernel_size: tuple[int, int],
    *,
    device: "torch.device",
    dtype: "torch.dtype | None" = None,
) -> "torch.Tensor":
    """Return the physical-coordinate ``D^{-1/2}`` Hankel normalization."""

    import torch

    multiplicity = multiplicity_map_torch(
        output_size, kernel_size, device=device
    ).clamp_min(1.0)
    if dtype is not None:
        multiplicity = multiplicity.to(dtype=dtype)
    return multiplicity.rsqrt()


def hankel_membership_consistency_np(
    physical_membership: np.ndarray,
    lifted_membership: np.ndarray,
    kernel_size: tuple[int, int],
) -> bool:
    """Check that Hankel duplicates retain their physical group labels."""

    physical = np.asarray(physical_membership)
    lifted = np.asarray(lifted_membership)
    if physical.ndim != 2:
        raise ValueError("physical_membership must have shape [H,W]")
    expected = block_hankel_np(physical[None, ...], kernel_size)
    if expected.shape != lifted.shape:
        raise ValueError(
            f"lifted_membership has shape {lifted.shape}, expected {expected.shape}"
        )
    return bool(np.array_equal(expected, lifted))


__all__ = [
    "block_hankel_np",
    "block_hankel_torch",
    "hankel_membership_consistency_np",
    "inverse_sqrt_multiplicity_map_torch",
    "multiplicity_map_np",
    "multiplicity_map_torch",
]
