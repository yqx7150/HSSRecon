"""Compact arithmetic helpers for the fixed-CG theory statement."""

from __future__ import annotations

import math
from collections.abc import Mapping


def cg_convergence_factor(condition_number: float) -> float:
    """Return the classical CG factor ``(sqrt(kappa)-1)/(sqrt(kappa)+1)``."""

    kappa = float(condition_number)
    if not math.isfinite(kappa) or kappa < 1.0:
        raise ValueError("condition_number must be finite and at least one")
    root = math.sqrt(kappa)
    return (root - 1.0) / (root + 1.0)


def psd_quadratic_spectral_bounds(
    ridge: float,
    term_upper_bounds: Mapping[str, float],
) -> dict[str, float]:
    """Compose conservative spectral bounds for a sum of PSD terms."""

    lower = float(ridge)
    if not math.isfinite(lower) or lower <= 0.0:
        raise ValueError("ridge must be finite and positive")
    if not isinstance(term_upper_bounds, Mapping):
        raise TypeError("term_upper_bounds must be a mapping")
    for name, value in term_upper_bounds.items():
        bound = float(value)
        if not math.isfinite(bound) or bound < 0.0:
            raise ValueError(f"term bound for {name!r} must be finite and non-negative")
    upper = lower + sum(float(value) for value in term_upper_bounds.values())
    return {
        "lambda_min_lower_bound": lower,
        "lambda_max_upper_bound": upper,
        "condition_number_upper_bound": upper / lower,
    }


__all__ = [
    "cg_convergence_factor",
    "psd_quadratic_spectral_bounds",
]
