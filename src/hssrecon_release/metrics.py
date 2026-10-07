"""Metrics matching the frozen fastMRI mainline evaluation protocol."""

from __future__ import annotations

import numpy as np


def center_crop_pair(
    reference: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the square center crop used by the release evaluation protocol."""

    reference = np.asarray(reference, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    if reference.shape != prediction.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {prediction.shape}")
    crop = min(reference.shape[-2:])
    row0 = (reference.shape[-2] - crop) // 2
    col0 = (reference.shape[-1] - crop) // 2
    return (
        reference[row0 : row0 + crop, col0 : col0 + crop],
        prediction[row0 : row0 + crop, col0 : col0 + crop],
    )


def benchmark_metrics(
    reference: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    """Return center-ROI PSNR, NMSE, and SSIM.

    This is intentionally kept in the release wrapper so the demo figure can
    be regenerated without importing the full reconstruction runner.  The
    formulas and foreground ROI are fixed by the release protocol.
    """

    reference, prediction = center_crop_pair(reference, prediction)
    data_range = float(reference.max())
    difference = prediction.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(difference**2))
    psnr = float(
        20.0 * np.log10(max(data_range, 1e-12))
        - 10.0 * np.log10(max(mse, 1e-20))
    )
    nmse = float(
        np.sum(difference**2)
        / max(float(np.sum(reference.astype(np.float64) ** 2)), 1e-30)
    )
    result: dict[str, float] = {"psnr": psnr, "nmse": nmse}
    try:
        from scipy import ndimage
        from skimage.metrics import structural_similarity

        global_ssim, ssim_map = structural_similarity(
            reference, prediction, data_range=data_range, full=True
        )
        roi = reference > 0.08 * (data_range + 1e-12)
        roi = ndimage.binary_closing(roi, iterations=2)
        roi = ndimage.binary_fill_holes(roi)
        roi = ndimage.binary_dilation(roi, iterations=3)
        roi_ssim = float(ssim_map[roi].mean()) if np.any(roi) else float(global_ssim)
        result.update(
            {
                "ssim": roi_ssim,
                "ssim_roi": roi_ssim,
                "ssim_global": float(global_ssim),
            }
        )
    except ImportError:
        # The demo can still render PSNR/NMSE in a minimal NumPy environment.
        pass
    return result
