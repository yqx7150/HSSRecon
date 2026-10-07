#!/usr/bin/env python3
"""Render the two fixed Poisson R6 fastMRI demonstration slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from hssrecon_release.metrics import benchmark_metrics, center_crop_pair  # noqa: E402


CASE_FILES = (
    "file_brain_AXFLAIR_200_6002467_slice008_poisson_R6.npz",
    "file_brain_AXT2_210_6001737_slice008_poisson_R6.npz",
)
METHODS = (
    ("rss_zf", "ZF"),
    ("rss_sake", "SAKE"),
    ("rss_ours", "HSSRecon"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=RELEASE_ROOT / "demo" / "data",
        help="directory containing the compact demonstration NPZ files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RELEASE_ROOT / "demo" / "outputs" / "poisson_r6_two_slice_montage.png",
        help="PNG path for the montage",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="optional JSON path; defaults beside --output",
    )
    return parser


def _load_case(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing demo case: {path}")
    with np.load(path) as loaded:
        required = {"mask", "rss_gt", *(key for key, _ in METHODS)}
        missing = sorted(required.difference(loaded.files))
        if missing:
            raise KeyError(f"{path} is missing keys: {', '.join(missing)}")
        return {key: np.asarray(loaded[key]) for key in required}


def _render(cases: list[tuple[str, dict[str, np.ndarray]]], output: Path) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = ("mask", "rss_gt", *(key for key, _ in METHODS))
    labels = ("Mask", "Reference", *(label for _, label in METHODS))
    figure, axes = plt.subplots(
        len(cases), len(columns), figsize=(16.0, 7.2), squeeze=False,
        constrained_layout=False,
    )
    metrics: dict[str, object] = {}

    for row, (case_name, data) in enumerate(cases):
        reference = np.asarray(data["rss_gt"], dtype=np.float32)
        _, reference_view = center_crop_pair(reference, reference)
        vmax = float(np.percentile(reference_view, 99.5))
        case_metrics: dict[str, object] = {}
        for column, (key, label) in enumerate(zip(columns, labels)):
            axis = axes[row, column]
            if key == "mask":
                image = center_crop_pair(data["mask"], data["mask"])[0]
                axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                axis.set_title("Mask" if row == 0 else "")
            else:
                image = np.asarray(data[key], dtype=np.float32)
                _, image_view = center_crop_pair(reference, image)
                axis.imshow(image_view, cmap="gray", vmin=0.0, vmax=vmax)
                if key != "rss_gt":
                    case_metrics[label] = benchmark_metrics(reference, image)
                    score = case_metrics[label]
                    title = f"{label}\n{score['psnr']:.2f} dB / SSIM {score.get('ssim', float('nan')):.3f}"
                else:
                    title = "Reference" if row == 0 else ""
                axis.set_title(title, fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

        display_name = case_name.replace("_poisson_R6", "").replace("file_brain_", "")
        axes[row, 0].set_ylabel(display_name, fontsize=10, rotation=90, labelpad=10)
        metrics[case_name] = case_metrics

    figure.suptitle(
        "HSSRecon · fastMRI-brain · Poisson R6 · fixed two-slice demonstration",
        fontsize=15,
        y=0.98,
    )
    figure.text(
        0.5,
        0.015,
        "Metrics: center-square ROI, matching the release evaluation protocol. "
        "These are descriptive examples, not a statistical sample.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.subplots_adjust(left=0.04, right=0.995, top=0.90, bottom=0.08, wspace=0.04, hspace=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)
    return {"output": str(output), "cases": metrics}


def main() -> None:
    args = _parser().parse_args()
    cases = [(path.name, _load_case(path)) for path in (args.data_root / name for name in CASE_FILES)]
    result = _render(cases, args.output)
    metrics_output = args.metrics_output or args.output.with_suffix(".json")
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote figure: {args.output}")
    print(f"wrote metrics: {metrics_output}")


if __name__ == "__main__":
    main()
