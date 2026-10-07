"""Evaluation metrics for magnitude MR images."""

from __future__ import annotations

import numpy as np

from .data import ifft2c_np, rss_np


def rss_from_kspace(kspace: np.ndarray) -> np.ndarray:
    return rss_np(ifft2c_np(kspace))
