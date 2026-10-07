"""Auditable contracts for physical-coordinate Hankel self-supervision.

The utilities in this module do not train a model.  They make the dependency
conditions behind a self-supervised claim explicit and serializable:

* masks are split on physical acquisition groups before Hankel lifting;
* the held-out loss is a physical-coordinate risk, not a duplicate-entry sum;
* a strict receptive-support audit is required when a local model can read a
  physical neighbourhood around its prediction.

The support condition is intentionally reported as ``incomplete`` when a
caller does not provide exact receptive supports.  Passing the mask/group
checks alone is therefore not evidence of a complete theorem for a particular
network.
"""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np
from .hankel import block_hankel_np, hankel_membership_consistency_np

def _bool_mask(value: np.ndarray, name: str, shape: tuple[int, int] | None=None) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f'{name} must have shape [H,W], got {mask.shape}')
    if shape is not None and mask.shape != shape:
        raise ValueError(f'{name} must have shape {shape}, got {mask.shape}')
    return mask

def _group_integrity(mask: np.ndarray, group_labels: np.ndarray, name: str) -> bool:
    """Return whether ``mask`` contains complete physical groups only."""
    labels = np.asarray(group_labels)
    selected = np.unique(labels[mask])
    selected = selected[selected >= 0]
    for group_id in selected:
        group = labels == group_id
        if not np.array_equal(mask[group], np.ones(int(group.sum()), dtype=bool)):
            raise ValueError(f'{name} splits physical acquisition group {int(group_id)}')
    return True

def audit_selection_observation_dependency(acquired_mask: np.ndarray, training_observation_mask: np.ndarray, selection_target_mask: np.ndarray, group_labels: np.ndarray | None=None) -> dict[str, Any]:
    """Audit the simpler reference-free dependency used by checkpoint selection.

    This check is distinct from a full receptive-field audit.  It asks whether
    any physical observation used by the training objective is also in the
    held-out ``Gamma`` target used for model selection.  A local model can still require
    the stronger support audit in :func:`audit_hankel_ssl_contract`.
    """
    acquired = _bool_mask(acquired_mask, 'acquired_mask')
    training = _bool_mask(training_observation_mask, 'training_observation_mask', acquired.shape)
    target = _bool_mask(selection_target_mask, 'selection_target_mask', acquired.shape)
    if np.any(training & ~acquired) or np.any(target & ~acquired):
        raise ValueError('training/selection dependency masks must be acquired subsets')
    overlap = training & target
    if group_labels is not None:
        labels = np.asarray(group_labels)
        if labels.shape != acquired.shape:
            raise ValueError('group_labels must match acquired_mask')
        _group_integrity(training, labels, 'training_observation')
        _group_integrity(target, labels, 'selection_target')
    return {'status': 'complete', 'passed': not bool(overlap.any()), 'training_observation_count': int(training.sum()), 'selection_target_count': int(target.sum()), 'overlap_count': int(overlap.sum()), 'reference_free_selection_dependency': not bool(overlap.any())}

def audit_receptive_support_dependency(source_masks: Sequence[np.ndarray], selection_target_mask: np.ndarray, *, support_mode: str) -> dict[str, Any]:
    """Audit the physical observations that can reach a selected target.

    ``source_masks`` contains every physical observation used either while
    fitting the parameters or while producing the validation prediction.  It
    is intentionally a union-of-dependencies audit: the validation input may
    reuse training observations, which is safe, but no source observation may
    be the held-out Gamma target itself.  This is the dependency condition
    needed by the conditional risk argument; requiring train and validation
    inputs to be disjoint would be unnecessarily strong for deployment-
    consistent validation.

    The caller must state how the support was obtained.  For the current
    factorized encoder, ``global_visible_input`` means that the exact source
    support is the union of visible physical k-space coordinates; mask-only
    geometry channels do not add noisy observations.
    """
    if not source_masks:
        raise ValueError('source_masks must contain at least one dependency mask')
    normalized = tuple((_bool_mask(mask, f'source_masks[{index}]') for (index, mask) in enumerate(source_masks)))
    shape = normalized[0].shape
    if any((mask.shape != shape for mask in normalized)):
        raise ValueError('all source_masks must share the same shape')
    target = _bool_mask(selection_target_mask, 'selection_target_mask', shape)
    source = np.zeros(shape, dtype=bool)
    for mask in normalized:
        source |= mask
    overlap = source & target
    return {'status': 'complete', 'provided': True, 'support_mode': str(support_mode), 'source_component_count': len(normalized), 'source_physical_coordinates': int(source.sum()), 'selection_target_coordinates': int(target.sum()), 'overlap_physical_coordinates': int(overlap.sum()), 'passed': not bool(overlap.any()), 'required_for_strict_claim': True, 'validation_input_may_reuse_training_support': True}

def audit_hankel_ssl_contract(acquired_mask: np.ndarray, input_mask: np.ndarray, target_mask: np.ndarray, validation_mask: np.ndarray, group_labels: np.ndarray, *, kernel_size: tuple[int, int] | None=None, training_observation_mask: np.ndarray | None=None, selection_target_mask: np.ndarray | None=None, validation_input_mask: np.ndarray | None=None, receptive_support_mode: str | None=None) -> dict[str, Any]:
    """Audit one physical input/target/validation draw.

    ``input_mask`` and ``target_mask`` are the masks used by one training
    view.  ``validation_mask`` is the fixed reference-free selection support.
    The function raises for malformed array geometry or a violated invariant;
    it returns a JSON-friendly report for valid geometry.  The returned
    ``strict_claim_ready`` flag is deliberately false unless exact receptive
    supports were supplied and found disjoint.

    """
    acquired = _bool_mask(acquired_mask, 'acquired_mask')
    shape = acquired.shape
    input_view = _bool_mask(input_mask, 'input_mask', shape)
    target = _bool_mask(target_mask, 'target_mask', shape)
    validation = _bool_mask(validation_mask, 'validation_mask', shape)
    labels = np.asarray(group_labels)
    if labels.shape != shape or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('group_labels must be an integer array matching acquired_mask')
    used = input_view | target | validation
    if np.any(used & ~acquired):
        raise ValueError('input, target, or validation contains an unacquired coordinate')
    if np.any(input_view & target):
        raise ValueError('input and held-out target masks overlap')
    if np.any(input_view & validation):
        raise ValueError('training input and validation masks overlap')
    if np.any(target & validation):
        raise ValueError('held-out target and validation masks overlap')
    if not input_view.any() or not target.any() or (not validation.any()):
        raise ValueError('input, target, and validation masks must all be non-empty')
    acquired_labels = labels[acquired]
    if np.any(labels[~acquired] != -1):
        raise ValueError('group_labels must be -1 outside acquired_mask')
    if acquired_labels.size == 0 or np.any(acquired_labels < 0):
        raise ValueError('every acquired coordinate needs a non-negative group id')
    group_ids = np.unique(acquired_labels)
    if not np.array_equal(group_ids, np.arange(group_ids.size, dtype=group_ids.dtype)):
        raise ValueError('group ids must be contiguous from zero')
    for (name, mask) in (('input', input_view), ('target', target), ('validation', validation)):
        _group_integrity(mask, labels, name)
    hankel_audit: dict[str, Any]
    if kernel_size is None:
        hankel_audit = {'status': 'not_requested', 'passed': None, 'kernel_size': None}
    else:
        if len(kernel_size) != 2 or min((int(value) for value in kernel_size)) <= 0:
            raise ValueError('kernel_size must contain two positive integers')
        lifted_labels = block_hankel_np(labels[None, ...], tuple((int(v) for v in kernel_size)))
        passed = bool(hankel_membership_consistency_np(labels, lifted_labels, tuple((int(v) for v in kernel_size))))
        if not passed:
            raise ValueError('Hankel lifting changed physical group membership')
        hankel_audit = {'status': 'complete', 'passed': True, 'kernel_size': [int(value) for value in kernel_size], 'duplicate_membership_preserved': True}
    if training_observation_mask is None and selection_target_mask is None:
        dependency = {'status': 'incomplete', 'provided': False, 'passed': None, 'reason': 'training/selection observation masks were not supplied'}
    elif training_observation_mask is None or selection_target_mask is None:
        raise ValueError('training_observation_mask and selection_target_mask must be supplied together')
    else:
        dependency = audit_selection_observation_dependency(acquired, training_observation_mask, selection_target_mask, labels)
    if validation_input_mask is None and receptive_support_mode is None:
        receptive_support = {'status': 'incomplete', 'provided': False, 'passed': None, 'reason': 'exact validation-input dependency support was not supplied', 'required_for_strict_claim': True}
    elif validation_input_mask is None or receptive_support_mode is None:
        raise ValueError('validation_input_mask and receptive_support_mode must be supplied together')
    elif training_observation_mask is None or selection_target_mask is None:
        raise ValueError('training_observation_mask and selection_target_mask are required for receptive support audit')
    else:
        receptive_support = audit_receptive_support_dependency((training_observation_mask, validation_input_mask), selection_target_mask, support_mode=str(receptive_support_mode))
    static_passed = bool(hankel_audit['passed'] is not False and (dependency['passed'] is not False) and (receptive_support['passed'] is not False))
    receptive_complete = bool(receptive_support['status'] == 'complete' and receptive_support['passed'])
    strict_ready = bool(static_passed and receptive_complete)
    return {'protocol': 'physical-group-quotient-hankel-v1', 'passed': static_passed, 'strict_claim_ready': strict_ready, 'risk_domain': 'physical_kspace', 'representation_domain': 'quotient_hankel', 'mask_split_stage': 'physical_groups_before_lifting', 'heldout_loss': 'physical_coordinate_ipw', 'multiplicity_correction': 'inverse_hankel_multiplicity_1_over_d', 'reference_free_training_and_selection': True, 'assumptions': ['positive_target_inclusion_probability', 'groupwise_sampling_protocol_is_known', 'support_disjoint_receptive_fields_for_local_models', 'noise_independence_across_heldout_physical_groups_for_risk_interpretation'], 'mask_counts': {'acquired': int(acquired.sum()), 'input': int(input_view.sum()), 'target': int(target.sum()), 'validation': int(validation.sum()), 'unused_acquired': int((acquired & ~used).sum())}, 'group_count': int(group_ids.size), 'hankel_duplicate_audit': hankel_audit, 'selection_observation_dependency': dependency, 'receptive_support_audit': receptive_support, 'claim_boundary': 'strict quotient-Hankel self-supervision is conditional on the receptive-support audit and the stated sampling/noise assumptions'}

def contract_summary(*, strict_receptive_support_audit: bool=False) -> dict[str, Any]:
    """Return stable method metadata for experiment artifacts and reports."""
    return {'protocol': 'physical-group-quotient-hankel-v1', 'risk_domain': 'physical_kspace', 'representation_domain': 'quotient_hankel', 'mask_split_stage': 'physical_groups_before_lifting', 'multiplicity_correction': 'inverse_hankel_multiplicity_1_over_d', 'data_consistency': 'hard_acquired_kspace_projection', 'reference_free_training_and_selection': True, 'strict_receptive_support_audit': bool(strict_receptive_support_audit), 'strict_receptive_support_audit_required_for_unqualified_claim': True}
