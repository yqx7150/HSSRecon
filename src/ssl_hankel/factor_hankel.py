"""Fixed factor-Hankel subspace model used by the HSSRecon demo release.

The module contains only the paper mainline used by the public demo: a
geometry-conditioned two-component Hankel subspace encoder, an explicit
multiplicity-normalized Hankel normal operator, and a hard-data-consistent
finite-step conjugate-gradient solver.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _stable_frame_torch(factor: 'torch.Tensor') -> 'torch.Tensor':
    """Build a regularized orthonormal frame without QR backpropagation."""
    import torch

    if factor.ndim != 3 or not torch.is_complex(factor):
        raise ValueError('factor must be complex with shape [B,F,Q]')
    gram = factor.mH @ factor
    scale = torch.real(torch.diagonal(gram, dim1=-2, dim2=-1)).mean(dim=-1)
    scale = scale.detach().clamp_min(1e-6)
    jitter = (1e-4 * scale)[:, None, None]
    eye = torch.eye(
        int(factor.shape[-1]), dtype=gram.dtype, device=factor.device
    )[None].expand(factor.shape[0], -1, -1)
    chol = torch.linalg.cholesky(gram + jitter.to(dtype=gram.real.dtype) * eye)
    inverse_chol = torch.linalg.solve_triangular(
        chol.mH, eye, upper=True
    )
    return factor @ inverse_chol


def hankel_adjoint_torch(
    lifted: 'torch.Tensor',
    output_size: tuple[int, int],
    kernel_size: tuple[int, int],
    num_coils: int,
) -> 'torch.Tensor':
    """Apply the Euclidean adjoint of the explicit block-Hankel lift."""
    import torch
    from torch.nn import functional

    if lifted.ndim != 3 or not torch.is_complex(lifted):
        raise ValueError('lifted must be complex with shape [B,P,F]')
    height, width = (int(output_size[0]), int(output_size[1]))
    kh, kw = (int(kernel_size[0]), int(kernel_size[1]))
    if min(height, width, kh, kw, int(num_coils)) <= 0:
        raise ValueError('output, kernel and coil dimensions must be positive')
    expected_rows = (height - kh + 1) * (width - kw + 1)
    expected_columns = int(num_coils) * kh * kw
    if kh > height or kw > width or tuple(lifted.shape[1:]) != (expected_rows, expected_columns):
        raise ValueError(
            f'lifted must have trailing shape {(expected_rows, expected_columns)}, '
            f'got {tuple(lifted.shape[1:])}'
        )
    columns = lifted.reshape(lifted.shape[0], expected_rows, int(num_coils), kh, kw)
    columns = columns.permute(0, 2, 3, 4, 1).reshape(
        lifted.shape[0], int(num_coils) * kh * kw, expected_rows
    )
    real = functional.fold(columns.real, (height, width), (kh, kw), stride=1)
    imag = functional.fold(columns.imag, (height, width), (kh, kw), stride=1)
    return torch.complex(real, imag)


def sampling_geometry_descriptor_torch(
    acquisition_mask: 'torch.Tensor', *, center_size: int = 24
) -> 'torch.Tensor':
    """Summarize acquisition geometry without reading k-space values."""
    import torch

    mask = acquisition_mask
    if mask.ndim == 2:
        mask = mask[None]
    if mask.ndim != 3:
        raise ValueError('acquisition_mask must have shape [B,H,W] or [H,W]')
    values = mask.to(dtype=torch.float32)
    _, height, width = values.shape
    side = int(center_size)
    if side <= 0:
        raise ValueError('center_size must be positive')
    center_height = min(side, int(height))
    center_width = min(side, int(width))
    row_start = (int(height) - center_height) // 2
    col_start = (int(width) - center_width) // 2
    center = values[:, row_start:row_start + center_height, col_start:col_start + center_width]
    row_density = values.mean(dim=-1)
    col_density = values.mean(dim=-2)

    def _std(value: 'torch.Tensor') -> 'torch.Tensor':
        return value.std(dim=-1, unbiased=False)

    row_delta = row_density[:, 1:] - row_density[:, :-1]
    col_delta = col_density[:, 1:] - col_density[:, :-1]
    return torch.stack(
        (
            values.mean(dim=(-2, -1)),
            center.mean(dim=(-2, -1)),
            _std(row_density),
            _std(col_density),
            row_delta.abs().mean(dim=-1),
            col_delta.abs().mean(dim=-1),
            row_density.amax(dim=-1),
            row_density.amin(dim=-1),
            col_density.amax(dim=-1),
            col_density.amin(dim=-1),
            center.mean(dim=-1).mean(dim=-1),
            center.mean(dim=-2).mean(dim=-1),
        ),
        dim=-1,
    )


def hankel_sampling_component_projectors_torch(
    factors: 'list[torch.Tensor] | tuple[torch.Tensor, ...]', *, orthogonalize: bool = True
) -> 'list[torch.Tensor]':
    """Build orthogonalized gauge-invariant projectors for the two components."""
    import torch

    if not isinstance(factors, (list, tuple)) or not factors:
        raise ValueError('factors must be a non-empty list or tuple')
    if any(value.ndim != 3 or not torch.is_complex(value) for value in factors):
        raise ValueError('factors must contain complex [B,F,Q] tensors')
    batch, feature_dim = factors[0].shape[:2]
    if any(tuple(value.shape[:2]) != (batch, feature_dim) for value in factors):
        raise ValueError('all factors must share [B,F]')

    projectors: list[torch.Tensor] = []
    occupied = None
    for factor in factors:
        if int(factor.shape[-1]) <= 0:
            raise ValueError('each factor must contain at least one atom')
        q = _stable_frame_torch(factor)
        if orthogonalize and occupied is not None:
            q = q - occupied @ (occupied.mH @ q)
            q = _stable_frame_torch(q)
        projector = q @ q.mH
        projectors.append(0.5 * (projector + projector.mH))
        if orthogonalize:
            occupied = q if occupied is None else torch.cat((occupied, q), dim=-1)
    return projectors


def hankel_sampling_mixture_projector_torch(
    factors: 'list[torch.Tensor] | tuple[torch.Tensor, ...]',
    weights: 'torch.Tensor',
    *,
    orthogonalize: bool = True,
) -> 'torch.Tensor':
    """Combine the structural and sampling-conditioned Hankel subspaces."""
    import torch

    projectors = hankel_sampling_component_projectors_torch(
        factors, orthogonalize=orthogonalize
    )
    batch = int(factors[0].shape[0])
    if weights.ndim == 1:
        weights = weights[None].expand(batch, -1)
    if tuple(weights.shape) != (batch, len(projectors)):
        raise ValueError('weights must have shape [B,S] or [S]')
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError('weights must be finite and non-negative')
    weights = weights.to(dtype=factors[0].real.dtype, device=factors[0].device)
    result = torch.zeros_like(projectors[0])
    for index, projector in enumerate(projectors):
        result = result + weights[:, index, None, None].to(result.dtype) * projector
    return result


def hankel_projector_torch(
    kspace: 'torch.Tensor', projector: 'torch.Tensor', kernel_size: tuple[int, int]
) -> 'torch.Tensor':
    """Apply a Hankel feature-space projector to physical k-space."""
    import torch
    from .hankel import block_hankel_torch

    if kspace.ndim != 4 or not torch.is_complex(kspace):
        raise ValueError('kspace must be complex [B,C,H,W]')
    if projector.ndim != 3 or not torch.is_complex(projector):
        raise ValueError('projector must be complex [B,F,F]')
    lifted = block_hankel_torch(kspace, kernel_size)
    expected = (int(lifted.shape[0]), int(lifted.shape[-1]), int(lifted.shape[-1]))
    if tuple(projector.shape) != expected:
        raise ValueError(f'projector shape is incompatible with kspace and kernel: expected {expected}')
    return lifted @ projector


def hankel_normal_operator_torch(
    kspace: 'torch.Tensor',
    projector: 'torch.Tensor',
    kernel_size: tuple[int, int],
) -> 'torch.Tensor':
    """Apply the explicit multiplicity-normalized Hankel normal operator."""
    import torch
    from .hankel import block_hankel_torch, inverse_sqrt_multiplicity_map_torch

    if kspace.ndim != 4 or not torch.is_complex(kspace):
        raise ValueError('kspace must be complex [B,C,H,W]')
    if projector.ndim != 3 or not torch.is_complex(projector):
        raise ValueError('projector must be complex [B,F,F]')
    if tuple(projector.shape[:1]) != tuple(kspace.shape[:1]):
        raise ValueError('projector batch and kspace batch must match')
    kh, kw = (int(kernel_size[0]), int(kernel_size[1]))
    if min(kh, kw) <= 0 or kh > int(kspace.shape[-2]) or kw > int(kspace.shape[-1]):
        raise ValueError('kernel_size must fit kspace')
    multiplicity_scale = inverse_sqrt_multiplicity_map_torch(
        (int(kspace.shape[-2]), int(kspace.shape[-1])),
        kernel_size,
        device=kspace.device,
        dtype=kspace.real.dtype,
    )
    transformed = kspace
    transformed = transformed * multiplicity_scale[None, None]
    lifted = block_hankel_torch(transformed, kernel_size)
    projected = lifted @ projector
    result = hankel_adjoint_torch(
        projected,
        output_size=(int(kspace.shape[-2]), int(kspace.shape[-1])),
        kernel_size=kernel_size,
        num_coils=int(kspace.shape[1]),
    )
    result = result * multiplicity_scale[None, None]
    return result / float(kh * kw)


def hankel_observability_torch(
    acquisition_mask: 'torch.Tensor', kernel_size: tuple[int, int]
) -> 'torch.Tensor':
    """Return mean local acquisition coverage of valid Hankel windows."""
    import torch
    from torch.nn import functional

    mask = acquisition_mask
    if mask.ndim == 2:
        mask = mask[None]
    if mask.ndim != 3:
        raise ValueError('acquisition_mask must have shape [B,H,W] or [H,W]')
    kh, kw = (int(kernel_size[0]), int(kernel_size[1]))
    if min(kh, kw) <= 0 or kh > int(mask.shape[-2]) or kw > int(mask.shape[-1]):
        raise ValueError('kernel_size must fit the acquisition mask')
    coverage = functional.avg_pool2d(mask.to(dtype=torch.float32)[:, None], (kh, kw), stride=1)
    return coverage.mean(dim=(-3, -2, -1)).clamp(min=0.0, max=1.0)


def hankel_nullspace_energy_torch(
    kspace: 'torch.Tensor', projector: 'torch.Tensor', kernel_size: tuple[int, int], *, normalize: bool = True
) -> 'torch.Tensor':
    """Measure reference-free energy in the predicted Hankel filter subspace."""
    from .hankel import block_hankel_torch

    filtered = hankel_projector_torch(kspace, projector, kernel_size)
    numerator = filtered.abs().square().mean()
    if not normalize:
        return numerator
    denominator = block_hankel_torch(kspace, kernel_size).abs().square().mean().clamp_min(1e-12)
    return numerator / denominator


def _masked_inner_product(left: 'torch.Tensor', right: 'torch.Tensor') -> 'torch.Tensor':
    import torch

    return torch.real(torch.sum(torch.conj(left) * right))


def solve_hankel_nullspace_proximal_torch(
    measured: 'torch.Tensor',
    dc_mask: 'torch.Tensor',
    projector: 'torch.Tensor',
    kernel_size: tuple[int, int],
    *,
    proximal_lambda: float = 1.0,
    ridge: float = 0.001,
    steps: int = 8,
    tolerance: float = 1e-6,
) -> 'torch.Tensor':
    """Solve the fixed hard-data-consistent Hankel quadratic by CG.

    During training the scalar CG coefficients are detached from the
    autograd graph.  The forward solve is unchanged, while this truncated
    differentiation avoids unstable gradients through long coefficient
    recurrences on high-dynamic-range k-space inputs.
    """
    import torch

    if measured.ndim != 4 or not torch.is_complex(measured):
        raise ValueError('measured must be complex [B,C,H,W]')
    if projector.ndim != 3 or not torch.is_complex(projector):
        raise ValueError('projector must be complex [B,F,F]')
    if dc_mask.ndim == 2:
        dc_mask = dc_mask[None]
    expected_mask = (int(measured.shape[0]), int(measured.shape[-2]), int(measured.shape[-1]))
    if tuple(dc_mask.shape) != expected_mask:
        raise ValueError(f'dc_mask must have shape {expected_mask}')
    if tuple(projector.shape[:1]) != tuple(measured.shape[:1]):
        raise ValueError('projector batch and measured batch must match')
    if float(proximal_lambda) < 0.0 or float(ridge) <= 0.0:
        raise ValueError('proximal_lambda must be non-negative and ridge positive')
    if int(steps) <= 0 or float(tolerance) < 0.0:
        raise ValueError('steps must be positive and tolerance must be non-negative')

    dc = dc_mask.bool()[:, None]
    free = ~dc
    measured_dc = torch.where(dc, measured, torch.zeros_like(measured))

    def regularizer(value: 'torch.Tensor') -> 'torch.Tensor':
        normal = hankel_normal_operator_torch(
            value,
            projector,
            kernel_size,
        )
        return free * (float(proximal_lambda) * normal + float(ridge) * value)

    checkpoint_enabled = torch.is_grad_enabled() and bool(projector.requires_grad)
    if checkpoint_enabled:
        from torch.utils.checkpoint import checkpoint

        def apply_regularizer(value: 'torch.Tensor') -> 'torch.Tensor':
            return checkpoint(regularizer, value, use_reentrant=False)
    else:
        def apply_regularizer(value: 'torch.Tensor') -> 'torch.Tensor':
            return regularizer(value)

    solution = torch.zeros_like(measured_dc)
    rhs = -apply_regularizer(measured_dc)
    residual = rhs - apply_regularizer(solution)
    residual_norm = _masked_inner_product(residual, residual)
    direction = residual
    krylov_norm = residual_norm
    for _ in range(int(steps)):
        applied = apply_regularizer(direction)
        denominator = _masked_inner_product(direction, applied).clamp_min(1e-6)
        alpha = (krylov_norm / denominator).detach()
        solution = solution + alpha * direction
        residual = residual - alpha * applied
        next_norm = _masked_inner_product(residual, residual)
        if bool((next_norm.detach() <= float(tolerance) ** 2).cpu()):
            break
        next_krylov_norm = _masked_inner_product(residual, residual)
        beta = (next_krylov_norm / krylov_norm.clamp_min(1e-6)).detach()
        direction = residual + beta * direction
        krylov_norm = next_krylov_norm

    result = measured_dc + free * solution
    return torch.where(dc, measured, result)


def subspace_consistency_loss_torch(
    projector_a: 'torch.Tensor', projector_b: 'torch.Tensor'
) -> 'torch.Tensor':
    """Gauge-invariant cross-view consistency for two learned subspaces."""
    import torch

    if projector_a.shape != projector_b.shape or projector_a.ndim != 3:
        raise ValueError('projectors must share shape [B,F,F]')
    if not torch.is_complex(projector_a) or not torch.is_complex(projector_b):
        raise ValueError('projectors must be complex')
    return (projector_a - projector_b).abs().square().mean()


def _factor_encoder_class():
    import torch
    from torch import nn

    class FactorHankelEncoder(nn.Module):
        """Geometry-conditioned two-component Hankel subspace encoder."""

        def __init__(
            self,
            *,
            num_coils: int,
            kernel_size: tuple[int, int] = (3, 3),
            filter_rank: int = 8,
            base_channels: int = 32,
            depth: int = 4,
            factor_stabilizer: float = 0.001,
            sampling_detail_weight: float = 0.1,
            sampling_detail_rank: int = 8,
        ) -> None:
            super().__init__()
            kh, kw = (int(kernel_size[0]), int(kernel_size[1]))
            feature_dim = int(num_coils) * kh * kw
            if min(int(num_coils), kh, kw, int(filter_rank), int(sampling_detail_rank)) <= 0:
                raise ValueError('coil, kernel and rank dimensions must be positive')
            if int(filter_rank) + int(sampling_detail_rank) > feature_dim:
                raise ValueError('combined sampling-aware rank exceeds the Hankel feature dimension')
            if int(base_channels) <= 0 or int(depth) <= 0:
                raise ValueError('base_channels and depth must be positive')
            if not 0.0 < float(factor_stabilizer) < 1.0:
                raise ValueError('factor_stabilizer must lie in (0, 1)')
            if not 0.0 < float(sampling_detail_weight) < 1.0:
                raise ValueError('sampling_detail_weight must lie in (0, 1)')

            self.num_coils = int(num_coils)
            self.kernel_size = (kh, kw)
            self.filter_rank = int(filter_rank)
            self.sampling_detail_rank = int(sampling_detail_rank)
            self.factor_stabilizer = float(factor_stabilizer)
            self.sampling_detail_weight = float(sampling_detail_weight)
            input_channels = 2 * int(num_coils) + 2
            layers: list[nn.Module] = [
                nn.Conv2d(input_channels, int(base_channels), 5, padding=2),
                nn.GELU(),
            ]
            for _ in range(int(depth) - 1):
                layers.extend(
                    [
                        nn.Conv2d(int(base_channels), int(base_channels), 3, padding=1),
                        nn.GroupNorm(1, int(base_channels)),
                        nn.GELU(),
                    ]
                )
            self.encoder = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.geometry_encoder = nn.Sequential(
                nn.Linear(12, int(base_channels)),
                nn.GELU(),
                nn.Linear(int(base_channels), int(base_channels)),
            )
            feature_ranks = [self.filter_rank, self.sampling_detail_rank]
            self.subspace_heads = nn.ModuleList(
                [
                    nn.Linear(int(base_channels), 2 * feature_dim * int(rank))
                    for rank in feature_ranks
                ]
            )
            self.weight_head = nn.Linear(int(base_channels), 2)
            self.confidence_head = nn.Linear(int(base_channels), 1)
            with torch.no_grad():
                self.weight_head.bias.zero_()
                self.confidence_head.bias.fill_(2.0)

        def forward(
            self,
            masked_kspace: 'torch.Tensor',
            input_mask: 'torch.Tensor',
            acquisition_mask: 'torch.Tensor',
        ) -> tuple[None, 'torch.Tensor']:
            if masked_kspace.ndim != 4 or not torch.is_complex(masked_kspace):
                raise ValueError('masked_kspace must be complex [B,C,H,W]')
            batch, coils, height, width = masked_kspace.shape
            if coils != self.num_coils:
                raise ValueError('masked_kspace coil count does not match encoder')
            if input_mask.ndim == 2:
                input_mask = input_mask[None]
            if acquisition_mask.ndim == 2:
                acquisition_mask = acquisition_mask[None]
            expected = (batch, height, width)
            if tuple(input_mask.shape) != expected or tuple(acquisition_mask.shape) != expected:
                raise ValueError('input_mask and acquisition_mask must be [B,H,W]')

            visible_kspace = masked_kspace * input_mask[:, None].to(dtype=masked_kspace.real.dtype)
            features = torch.cat(
                (
                    visible_kspace.real,
                    visible_kspace.imag,
                    input_mask[:, None].to(masked_kspace.real.dtype),
                    acquisition_mask[:, None].to(masked_kspace.real.dtype),
                ),
                dim=1,
            )
            embedding = self.pool(self.encoder(features)).flatten(1)
            embedding = embedding + self.geometry_encoder(
                sampling_geometry_descriptor_torch(acquisition_mask)
            )

            logits = self.weight_head(embedding)
            structural_weight = torch.ones(batch, 1, dtype=embedding.dtype, device=embedding.device)
            detail_weight = self.sampling_detail_weight * torch.sigmoid(logits[:, 1:])
            detail_weight = detail_weight * torch.sigmoid(self.confidence_head(embedding))
            weights = torch.cat((structural_weight, detail_weight), dim=-1)

            feature_dim = self.num_coils * self.kernel_size[0] * self.kernel_size[1]
            factors: list[torch.Tensor] = []
            for head, rank in zip(self.subspace_heads, (self.filter_rank, self.sampling_detail_rank)):
                raw = head(embedding).reshape(batch, feature_dim, int(rank), 2)
                factor = torch.complex(raw[..., 0], raw[..., 1])
                frame = torch.zeros_like(factor)
                diagonal = torch.arange(int(rank), device=factor.device)
                frame[:, diagonal, diagonal] = 1.0
                factors.append(factor + float(self.factor_stabilizer) * frame)
            projector = hankel_sampling_mixture_projector_torch(
                factors, weights, orthogonalize=True
            )
            return None, projector

    return FactorHankelEncoder


def make_factor_hankel_encoder(**kwargs):
    """Construct the fixed demo encoder without importing Torch eagerly."""
    return _factor_encoder_class()(**kwargs)


__all__ = [
    'hankel_adjoint_torch',
    'sampling_geometry_descriptor_torch',
    'hankel_sampling_component_projectors_torch',
    'hankel_sampling_mixture_projector_torch',
    'hankel_projector_torch',
    'hankel_normal_operator_torch',
    'hankel_observability_torch',
    'hankel_nullspace_energy_torch',
    'solve_hankel_nullspace_proximal_torch',
    'subspace_consistency_loss_torch',
    'make_factor_hankel_encoder',
]
