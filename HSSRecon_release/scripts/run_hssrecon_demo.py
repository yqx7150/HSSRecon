#!/usr/bin/env python3
"""Run the frozen HSSRecon per-slice demo on one or more carrier NPZ inputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parents[1]
RUNNER = RELEASE_ROOT / "scripts" / "hssrecon_demo_runner.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path, dest="inputs")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--max-coils", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--inputs",
        *(str(path) for path in args.inputs),
        "--output-root",
        str(args.output_root),
        "--steps",
        str(args.steps),
        "--device",
        args.device,
        "--seed",
        "83",
        "--mask-seed",
        "89",
        "--crop-size",
        "640",
        "320",
        "--max-coils",
        str(args.max_coils),
    ]
    return command


def main() -> None:
    args = _parser().parse_args()
    for path in args.inputs:
        if not path.is_file():
            raise FileNotFoundError(f"missing input: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            missing = sorted({"kspace", "mask"}.difference(loaded.files))
        if missing:
            raise SystemExit(
                f"{path} is a visual snapshot, not a reconstruction carrier; "
                "provide a carrier NPZ containing kspace and mask "
                f"(missing: {', '.join(missing)})"
            )
    command = _command(args)
    print(" ".join(command))
    if args.dry_run:
        return
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for reconstruction. Install the release "
            "extra with: python -m pip install -e \".[reconstruction]\""
        ) from exc
    env = os.environ.copy()
    source_path = str(RELEASE_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_path, env.get("PYTHONPATH")) if value
    )
    subprocess.run(command, check=True, cwd=RELEASE_ROOT, env=env)


if __name__ == "__main__":
    main()
