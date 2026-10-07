"""Load the compact multi-coil NPZ carrier used by the demo."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class KspaceSample:
    """Normalized k-space and optional post-lock reference for one slice."""

    name: str
    measured_kspace: np.ndarray
    mask: np.ndarray
    full_kspace: np.ndarray | None
    sampling_kind: str
    requested_acceleration: int | None


def infer_sampling_kind(path: str | Path) -> str:
    """Infer the sampling family from a carrier filename."""

    name = Path(path).name.lower()
    return "poisson" if re.search(r"(?:^|_)poisson(?:_|\.)", name) else "unknown"


def infer_requested_acceleration(path: str | Path) -> int | None:
    """Infer an ``R<number>`` acceleration token from a filename."""

    match = re.search(r"(?:^|_)R(\d+)(?:_|\.|$)", Path(path).name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def fft2c_np(image: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(image, axes=(-2, -1)),
            axes=(-2, -1),
            norm="ortho",
        ),
        axes=(-2, -1),
    )


def ifft2c_np(kspace: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace, axes=(-2, -1)),
            axes=(-2, -1),
            norm="ortho",
        ),
        axes=(-2, -1),
    )


def rss_np(coil_images: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0)).astype(np.float32)


def load_carrier_npz(
    path: str | Path,
    normalization_percentile: float = 99.0,
    *,
    include_reference: bool = True,
) -> KspaceSample:
    """Load a carrier with ``kspace`` and ``mask`` arrays.

    ``image_coils`` is read only when ``include_reference`` is true.  The
    runner therefore trains and selects its checkpoint without loading a
    reference reconstruction target.
    """

    sample_path = Path(path)
    with np.load(sample_path, allow_pickle=False) as loaded:
        required = {"kspace", "mask"}
        missing = required.difference(loaded.files)
        if missing:
            raise KeyError(f"{sample_path} is missing required arrays: {sorted(missing)}")
        measured = np.asarray(loaded["kspace"], dtype=np.complex64)
        mask = np.asarray(loaded["mask"], dtype=bool)
        image_coils = (
            np.asarray(loaded["image_coils"], dtype=np.complex64)
            if include_reference and "image_coils" in loaded.files
            else None
        )
        requested_acceleration = (
            int(np.asarray(loaded["requested_acceleration"]).item())
            if "requested_acceleration" in loaded.files
            else infer_requested_acceleration(sample_path)
        )
        sampling_kind = (
            str(np.asarray(loaded["sampling_kind"]).item()).lower()
            if "sampling_kind" in loaded.files
            else infer_sampling_kind(sample_path)
        )

    if measured.ndim != 3:
        raise ValueError(f"kspace must have shape [C,H,W], got {measured.shape}")
    if mask.shape != measured.shape[-2:]:
        raise ValueError(f"mask shape {mask.shape} does not match kspace {measured.shape}")

    measured = measured * mask[None, ...]
    zero_fill = rss_np(ifft2c_np(measured))
    scale = float(np.percentile(zero_fill, normalization_percentile))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.sqrt(np.mean(np.abs(measured[:, mask]) ** 2)) + 1e-8)

    full_kspace = None
    if image_coils is not None:
        if image_coils.ndim != 3 or image_coils.shape[:2] != mask.shape:
            raise ValueError(
                "image_coils must have shape [H,W,C] compatible with mask, "
                f"got {image_coils.shape}"
            )
        normalized_images = image_coils / scale
        full_kspace = fft2c_np(np.moveaxis(normalized_images, -1, 0)).astype(
            np.complex64
        )

    return KspaceSample(
        name=sample_path.stem,
        measured_kspace=(measured / scale).astype(np.complex64),
        mask=mask,
        full_kspace=full_kspace,
        sampling_kind=sampling_kind,
        requested_acceleration=requested_acceleration,
    )


__all__ = [
    "KspaceSample",
    "fft2c_np",
    "ifft2c_np",
    "infer_requested_acceleration",
    "infer_sampling_kind",
    "load_carrier_npz",
    "rss_np",
]
