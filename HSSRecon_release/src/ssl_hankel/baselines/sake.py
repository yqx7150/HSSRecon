"""Small independent SAKE comparison baseline used by the demo."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def sake_kspace_completion(
    kspace: torch.Tensor,
    mask: torch.Tensor,
    *,
    kernel_size: tuple[int, int] = (6, 6),
    rank_factor: float = 1.5,
    iterations: int = 30,
    verbose: bool = True,
) -> torch.Tensor:
    """Alternating low-rank Hankel projection and measured-data consistency."""
    if kspace.ndim != 3 or mask.shape != kspace.shape[-2:]:
        raise ValueError("Expected kspace [C,H,W] and compatible mask [H,W]")
    num_coils, height, width = kspace.shape
    kh, kw = kernel_size
    patch_rows = height - kh + 1
    patch_cols = width - kw + 1
    num_patches = patch_rows * patch_cols
    feature_dim = num_coils * kh * kw
    rank = min(feature_dim, max(1, math.floor(rank_factor * kh * kw)))
    measured = kspace.clone()
    reconstructed = kspace.to(torch.complex64).clone()
    mask = mask.bool().to(kspace.device)
    norm_input = torch.ones(
        1, feature_dim, num_patches, device=kspace.device, dtype=kspace.real.dtype
    )
    norm = F.fold(norm_input, (height, width), (kh, kw)).clamp_min(1.0)

    for iteration in range(iterations):
        matrix = (
            reconstructed.unfold(1, kh, 1)
            .unfold(2, kw, 1)
            .permute(1, 2, 0, 3, 4)
            .reshape(num_patches, feature_dim)
        )
        if num_patches > feature_dim:
            gram = matrix.mH @ matrix
            _, eigenvectors = torch.linalg.eigh(gram)
            basis = eigenvectors[:, -rank:]
            low_rank = (matrix @ basis) @ basis.mH
        else:
            left, values, right_h = torch.linalg.svd(matrix, full_matrices=False)
            low_rank = (left[:, :rank] * values[:rank]) @ right_h[:rank]
        columns = low_rank.T.unsqueeze(0)
        real = F.fold(columns.real, (height, width), (kh, kw)) / norm
        imag = F.fold(columns.imag, (height, width), (kh, kw)) / norm
        estimate = torch.complex(real, imag).squeeze(0)
        reconstructed = torch.where(mask[None], measured, estimate)
        if verbose and (iteration == 0 or (iteration + 1) % 10 == 0):
            print(f"SAKE iteration {iteration + 1}/{iterations}", flush=True)
    return reconstructed.to(kspace.dtype)
