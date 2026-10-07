"""Reference-free scan-specific mask splits and validation helpers.

The scan-specific protocol splits acquired *physical* sampling groups before
Hankel lifting.  This module deliberately contains only NumPy utilities so
that the leakage and partition audits can run in the lightweight local
environment without importing Torch.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from typing import Any
import numpy as np
from .hankel import block_hankel_np, hankel_membership_consistency_np

def _as_bool_mask(mask: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError(f'{name} must have shape [H,W], got {value.shape}')
    return value

def _point_group_labels_np(acquired_mask: np.ndarray) -> np.ndarray:
    """Represent every acquired coordinate with a compact group-label map."""
    acquired = _as_bool_mask(acquired_mask, 'acquired_mask')
    coordinates = np.argwhere(acquired)
    if coordinates.size == 0:
        raise ValueError('acquired_mask must contain at least one sample')
    labels = np.full(acquired.shape, -1, dtype=np.int32)
    labels[coordinates[:, 0], coordinates[:, 1]] = np.arange(coordinates.shape[0], dtype=np.int32)
    return labels

def _validate_group_partition(acquired_mask: np.ndarray, group_labels: np.ndarray, *, name: str='group_labels') -> None:
    acquired = _as_bool_mask(acquired_mask, 'acquired_mask')
    labels = np.asarray(group_labels)
    if labels.ndim != 2 or labels.shape != acquired.shape:
        raise ValueError(f'{name} must have shape {acquired.shape}, got {labels.shape}')
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f'{name} must contain integer group ids')
    ids = np.unique(labels[acquired])
    if ids.size < 2:
        raise ValueError(f'{name} must contain at least two non-empty groups')
    if np.any(labels[~acquired] != -1):
        raise ValueError(f'{name} labels unacquired coordinates')
    expected = np.arange(ids.size, dtype=ids.dtype)
    if not np.array_equal(ids, expected):
        raise ValueError(f'{name} ids must be contiguous from zero')

def _mask_from_group_indices(group_labels: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        raise ValueError('a mask split cannot contain zero groups')
    labels = np.asarray(group_labels)
    selected = np.zeros(int(np.max(labels)) + 1, dtype=bool)
    selected[np.asarray(indices, dtype=np.int64)] = True
    mask = np.zeros(labels.shape, dtype=bool)
    acquired = labels >= 0
    mask[acquired] = selected[labels[acquired]]
    return mask

def build_validation_group_folds(split: 'ScanMaskSplit', fold_count: int=2) -> tuple[tuple[int, ...], ...]:
    """Partition ``G_val`` into deterministic small held-out group folds.

    The folds are used only for deployment-consistent self-validation.  A
    validation call hides one fold ``F_i`` but leaves the other acquired
    validation groups visible, so its input operator is ``G_acq \\ F_i`` and
    is much closer to the final acquired-data inference operator than the
    original ``G_train``-only check.
    """
    if isinstance(fold_count, bool) or int(fold_count) <= 0:
        raise ValueError('fold_count must be a positive integer')
    validation_ids = np.asarray(split.validation_group_ids, dtype=np.int64)
    if validation_ids.size == 0:
        raise ValueError('split must contain at least one validation group')
    if np.unique(validation_ids).size != validation_ids.size:
        raise ValueError('validation_group_ids must be unique')
    requested = min(int(fold_count), int(validation_ids.size))
    folds = np.array_split(validation_ids, requested)
    return tuple((tuple((int(value) for value in fold.tolist())) for fold in folds if fold.size))

def _mask_digest(mask: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    return hashlib.sha256(value.tobytes()).hexdigest()

def _coordinates(mask: np.ndarray) -> list[list[int]]:
    return np.argwhere(mask).astype(np.int64).tolist()

@dataclass(frozen=True)
class ScanMaskSplit:
    """A deterministic group-level split for one acquired scan."""
    acquired_mask: np.ndarray
    validation_mask: np.ndarray
    train_mask: np.ndarray
    view1_mask: np.ndarray
    view2_mask: np.ndarray
    group_labels: np.ndarray
    validation_group_ids: tuple[int, ...]
    train_group_ids: tuple[int, ...]
    view1_group_ids: tuple[int, ...]
    view2_group_ids: tuple[int, ...]
    sampling_kind: str
    mask_split_seed: int
    validation_fraction: float
    view1_fraction: float

    def metadata(self, *, hankel_kernel_size: tuple[int, int] | None=None) -> dict[str, Any]:
        """Return JSON-serializable split metadata with physical coordinates."""
        masks = {'acquired': self.acquired_mask, 'validation': self.validation_mask, 'train': self.train_mask, 'view1': self.view1_mask, 'view2': self.view2_mask}
        audit = audit_scan_mask_split(self)
        if hankel_kernel_size is not None:
            audit['hankel_duplicate_membership'] = audit_hankel_membership(self, hankel_kernel_size)
        return {'protocol': 'scan-specific-mask-split-v1', 'sampling_kind': self.sampling_kind, 'grouping': 'one-acquired-coordinate-per-physical-group', 'mask_split_seed': int(self.mask_split_seed), 'validation_fraction': float(self.validation_fraction), 'view1_fraction': float(self.view1_fraction), 'group_count': int(np.max(self.group_labels) + 1), 'group_sizes': [int(value) for value in np.bincount(self.group_labels[self.group_labels >= 0].ravel(), minlength=int(np.max(self.group_labels) + 1))], 'validation_group_ids': list(self.validation_group_ids), 'train_group_ids': list(self.train_group_ids), 'view1_group_ids': list(self.view1_group_ids), 'view2_group_ids': list(self.view2_group_ids), 'split_counts': {name: int(value.sum()) for (name, value) in masks.items()}, 'mask_sha256': {name: _mask_digest(value) for (name, value) in masks.items()}, 'physical_coordinates': {name: _coordinates(value) for (name, value) in masks.items()}, 'audit': audit}

def build_scan_mask_split(acquired_mask: np.ndarray, *, sampling_kind: str, validation_fraction: float=0.2, view1_fraction: float=0.5, mask_split_seed: int=0, allow_empty_validation: bool=False) -> ScanMaskSplit:
    """Build a physical group split for one scan without reading a reference.

    ``G_acq`` is first partitioned into ``G_val`` and ``G_train``.  The latter
    is then partitioned into two disjoint self-supervised views.  Each
    acquired point is one physical group for the Poisson demo.
    """
    acquired = _as_bool_mask(acquired_mask, 'acquired_mask')
    if bool(allow_empty_validation):
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError('validation_fraction must be in [0, 1)')
    elif not 0.0 < validation_fraction < 1.0:
        raise ValueError('validation_fraction must be in (0, 1)')
    if not 0.0 < view1_fraction < 1.0:
        raise ValueError('view1_fraction must be in (0, 1)')
    group_labels = _point_group_labels_np(acquired)
    _validate_group_partition(acquired, group_labels)
    generator = np.random.default_rng(mask_split_seed)
    group_count = int(np.max(group_labels) + 1)
    order = generator.permutation(group_count)
    if bool(allow_empty_validation) and float(validation_fraction) == 0.0:
        validation_count = 0
    else:
        validation_count = int(np.floor(group_count * validation_fraction))
        validation_count = min(group_count - 1, max(1, validation_count))
    validation_ids = np.sort(order[:validation_count])
    train_ids = np.sort(order[validation_count:])
    if train_ids.size < 2:
        raise ValueError('scan-specific split needs at least two training groups after validation split')
    view1_count = int(np.floor(train_ids.size * view1_fraction))
    view1_count = min(train_ids.size - 1, max(1, view1_count))
    train_order = generator.permutation(train_ids)
    view1_ids = np.sort(train_order[:view1_count])
    view2_ids = np.sort(train_order[view1_count:])
    split = ScanMaskSplit(acquired_mask=acquired.copy(), validation_mask=np.zeros_like(acquired, dtype=bool) if validation_ids.size == 0 else _mask_from_group_indices(group_labels, validation_ids), train_mask=_mask_from_group_indices(group_labels, train_ids), view1_mask=_mask_from_group_indices(group_labels, view1_ids), view2_mask=_mask_from_group_indices(group_labels, view2_ids), group_labels=group_labels.copy(), validation_group_ids=tuple((int(value) for value in validation_ids)), train_group_ids=tuple((int(value) for value in train_ids)), view1_group_ids=tuple((int(value) for value in view1_ids)), view2_group_ids=tuple((int(value) for value in view2_ids)), sampling_kind=str(sampling_kind), mask_split_seed=int(mask_split_seed), validation_fraction=float(validation_fraction), view1_fraction=float(view1_fraction))
    audit_scan_mask_split(split)
    return split

def audit_scan_mask_split(split: ScanMaskSplit) -> dict[str, Any]:
    """Check physical disjointness, group membership and support preservation."""
    acquired = split.acquired_mask
    masks = {'validation': split.validation_mask, 'view1': split.view1_mask, 'view2': split.view2_mask}
    if any((mask.shape != acquired.shape for mask in masks.values())):
        raise ValueError('all split masks must match acquired_mask shape')
    if np.any(split.validation_mask & split.train_mask):
        raise ValueError('validation and train masks overlap')
    if np.any(split.view1_mask & split.view2_mask):
        raise ValueError('self-supervised views overlap')
    if not np.array_equal(split.view1_mask | split.view2_mask, split.train_mask):
        raise ValueError('view masks do not reconstruct train mask')
    if not np.array_equal(split.validation_mask | split.train_mask, acquired):
        raise ValueError('validation and train masks do not reconstruct acquired mask')
    if any((np.any(mask & ~acquired) for mask in masks.values())):
        raise ValueError('split mask contains an unacquired physical coordinate')
    _validate_group_partition(acquired, split.group_labels)
    selected = {'validation': set(split.validation_group_ids), 'view1': set(split.view1_group_ids), 'view2': set(split.view2_group_ids)}
    group_count = int(np.max(split.group_labels) + 1)
    if set().union(*selected.values()) != set(range(group_count)):
        raise ValueError('split group ids do not cover all acquired groups')
    if sum((len(value) for value in selected.values())) != group_count:
        raise ValueError('split group ids overlap')
    return {'passed': True, 'group_partition_disjoint': True, 'group_partition_exact': True, 'validation_train_disjoint': True, 'view1_view2_disjoint': True, 'support_exact': True, 'acquired_count': int(acquired.sum()), 'validation_count': int(split.validation_mask.sum()), 'train_count': int(split.train_mask.sum()), 'view1_count': int(split.view1_mask.sum()), 'view2_count': int(split.view2_mask.sum())}

def audit_hankel_membership(split: ScanMaskSplit, kernel_size: tuple[int, int]) -> dict[str, Any]:
    """Verify that the compact physical group labels survive Hankel lifting."""
    if len(kernel_size) != 2 or min(kernel_size) <= 0:
        raise ValueError('kernel_size must contain two positive integers')
    lifted = block_hankel_np(split.group_labels[None, ...], kernel_size)
    passed = hankel_membership_consistency_np(split.group_labels, lifted, kernel_size)
    if not passed:
        raise ValueError('Hankel lifting changed physical group membership')
    return {'passed': True, 'kernel_size': [int(value) for value in kernel_size], 'lifted_shape': [int(value) for value in lifted.shape], 'duplicate_membership_preserved': True}
