"""Run the fixed HSSRecon demonstration protocol on one or more carriers.

The public runner intentionally exposes only the paper mainline: physical-group
self-supervision before Hankel lifting, duplicate-aware full-2D Hankel
subspaces, reference-free validation/selection, a fixed constrained CG solve,
and hard consistency on acquired data.  Reference arrays are loaded only after
the checkpoint is locked, for post-lock metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_RELEASE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_RELEASE_SRC) not in sys.path:
    sys.path.insert(0, str(_RELEASE_SRC))

from ssl_hankel.data import load_carrier_npz
from ssl_hankel.factor_hankel import (
    hankel_nullspace_energy_torch,
    hankel_observability_torch,
    make_factor_hankel_encoder,
    solve_hankel_nullspace_proximal_torch,
    subspace_consistency_loss_torch,
)
from ssl_hankel.losses import physical_ipw_heldout_sum
from ssl_hankel.metrics import rss_from_kspace
from ssl_hankel.scan_specific import build_scan_mask_split, build_validation_group_folds
from ssl_hankel.ssl_contract import audit_hankel_ssl_contract, contract_summary
from ssl_hankel.theory_bounds import cg_convergence_factor, psd_quadratic_spectral_bounds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed HSSRecon demonstration protocol.")
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--mask-seed", type=int, default=89)
    parser.add_argument(
        "--crop-size",
        nargs=2,
        type=int,
        default=(640, 320),
        metavar=("HEIGHT", "WIDTH"),
        help="center crop used by the published demo protocol",
    )
    parser.add_argument("--max-coils", type=int, default=20, help="maximum number of coils retained from each input")
    return parser


def _center_crop(
    kspace: np.ndarray,
    mask: np.ndarray,
    crop_size: tuple[int, int],
    max_coils: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    height, width = (int(crop_size[0]), int(crop_size[1]))
    if min(height, width, int(max_coils)) <= 0:
        raise ValueError("crop-size and max-coils must be positive")
    if height > kspace.shape[-2] or width > kspace.shape[-1]:
        raise ValueError("crop-size must fit the input")
    row_start = (kspace.shape[-2] - height) // 2
    column_start = (kspace.shape[-1] - width) // 2
    bounds = (row_start, row_start + height, column_start, column_start + width)
    return (
        kspace[:max_coils, row_start : row_start + height, column_start : column_start + width].copy(),
        mask[row_start : row_start + height, column_start : column_start + width].copy(),
        bounds,
    )


def _device(value: str):
    import torch

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _scalar(value) -> float:
    return float(value.detach().cpu())


def _score_post_lock(
    sample_path: Path,
    output_kspace: np.ndarray,
    sake_kspace: np.ndarray,
    crop_bounds: tuple[int, int, int, int],
    *,
    normalization_percentile: float = 99.0,
) -> dict[str, dict[str, float]]:
    from hssrecon_release.metrics import benchmark_metrics

    row_start, row_stop, column_start, column_stop = crop_bounds
    reference_sample = load_carrier_npz(
        sample_path,
        normalization_percentile=float(normalization_percentile),
        include_reference=True,
    )
    if reference_sample.full_kspace is None:
        return {}
    if (row_start, row_stop, column_start, column_stop) != (
        0,
        reference_sample.mask.shape[0],
        0,
        reference_sample.mask.shape[1],
    ):
        return {"protocol_warning": "center-frequency crop; image metrics are deferred to full-FOV gate"}
    reference_kspace = np.asarray(reference_sample.full_kspace, dtype=np.complex64)
    reference = rss_from_kspace(reference_kspace[:, row_start:row_stop, column_start:column_stop])
    measured = reference_sample.measured_kspace[:, row_start:row_stop, column_start:column_stop]
    zero_fill = rss_from_kspace(measured)
    return {
        "zero_fill": benchmark_metrics(reference, zero_fill),
        "ours": benchmark_metrics(reference, rss_from_kspace(output_kspace)),
        "sake": benchmark_metrics(reference, rss_from_kspace(sake_kspace)),
    }


def _run_case(args, input_path: Path, output_root: Path) -> dict[str, object]:
    import torch

    kernel_size = (3, 3)
    torch.manual_seed(int(args.seed))

    sample = load_carrier_npz(input_path, normalization_percentile=99.0, include_reference=False)
    mask_full = np.asarray(sample.mask, dtype=bool)
    source_kspace = np.asarray(sample.measured_kspace, dtype=np.complex64).copy()
    cropped_kspace, cropped_mask, crop_bounds = _center_crop(
        source_kspace,
        mask_full,
        tuple(int(value) for value in args.crop_size),
        min(int(args.max_coils), int(source_kspace.shape[0])),
    )

    filter_rank = 16
    feature_dim = int(cropped_kspace.shape[0]) * 3 * 3
    if filter_rank + 8 > feature_dim:
        raise ValueError(
            "the demo protocol requires at least 24 Hankel features "
            f"for rank 16 + detail rank 8; got {feature_dim}"
        )
    device = _device(args.device)
    measured = torch.as_tensor(cropped_kspace, dtype=torch.complex64, device=device)[None]

    split = build_scan_mask_split(
        cropped_mask,
        sampling_kind=sample.sampling_kind,
        validation_fraction=0.2,
        view1_fraction=0.5,
        mask_split_seed=int(args.mask_seed),
    )
    validation_group_folds = build_validation_group_folds(split, fold_count=2)
    validation_fold_masks_np = tuple(
        split.validation_mask & np.isin(split.group_labels, np.asarray(group_ids, dtype=np.int64))
        for group_ids in validation_group_folds
    )
    acquired = torch.as_tensor(split.acquired_mask, dtype=torch.bool, device=device)[None]

    from ssl_hankel.baselines.sake import sake_kspace_completion

    model_kwargs = {
        "num_coils": int(measured.shape[1]),
        "kernel_size": (3, 3),
        "filter_rank": filter_rank,
        "base_channels": 16,
        "depth": 3,
        "sampling_detail_weight": 0.1,
        "sampling_detail_rank": 8,
    }
    model = make_factor_hankel_encoder(**model_kwargs).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("no trainable parameters remain after freezing the Hankel branch")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=0.0005, weight_decay=1e-6)
    generator = np.random.default_rng(int(args.mask_seed) + 1)

    train_coordinates = np.argwhere(split.train_mask)
    train_count = int(len(train_coordinates))
    target_count = min(train_count - 1, max(1, int(np.floor(train_count * 0.25))))
    schedule = np.zeros((2 * int(args.steps), *split.train_mask.shape), dtype=bool)
    for index in range(schedule.shape[0]):
        chosen = generator.choice(train_count, size=target_count, replace=False)
        schedule[index, train_coordinates[chosen, 0], train_coordinates[chosen, 1]] = True
    point_schedule = torch.as_tensor(schedule, dtype=torch.bool, device=device)
    point_input_mask = torch.as_tensor(split.train_mask, dtype=torch.bool, device=device)
    point_probability = torch.zeros(split.train_mask.shape, dtype=torch.float32, device=device)
    point_probability[point_input_mask] = float(target_count / train_count)

    history: list[dict[str, float | int]] = []
    validation_history: list[dict[str, float | int | str | list[float]]] = []
    contract_audits: list[dict[str, object]] = []
    observability_records: list[dict[str, float | str]] = []
    best_state = None
    best_validation = 1e309
    bad_intervals = 0
    start_time = time.time()

    def draw_tensor(draw_index: int):
        target_mask = point_schedule[int(draw_index) % int(point_schedule.shape[0])]
        input_mask = point_input_mask & ~target_mask
        result = (input_mask[None], target_mask, point_probability)
        if not contract_audits:
            input_np = result[0][0].detach().cpu().numpy().astype(bool)
            target_np = result[1].detach().cpu().numpy().astype(bool)
            contract_audits.append(
                audit_hankel_ssl_contract(
                    split.acquired_mask,
                    input_np,
                    target_np,
                    split.validation_mask,
                    split.group_labels,
                    kernel_size=(3, 3),
                    training_observation_mask=split.train_mask.copy(),
                    selection_target_mask=split.validation_mask,
                    validation_input_mask=split.acquired_mask & ~split.validation_mask,
                    receptive_support_mode="global_visible_input",
                )
            )
        return result

    def predict_projector(masked: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        _, projector = model(masked, input_mask, acquired)
        return projector

    def proximal_lambda_for(mask: torch.Tensor, *, label: str) -> float:
        coverage = hankel_observability_torch(mask, (3, 3))
        observability_records.append(
            {
                "label": str(label),
                "coverage": float(coverage.mean().detach().cpu()),
                "proximal_lambda": 1.0,
            }
        )
        return 1.0

    training_solver_steps = 8

    def theory_spectral_bound_metadata(
        training_steps: int, inference_steps: int
    ) -> dict[str, object]:
        ridge = 0.001
        observed_lambdas = [1.0]
        observed_lambdas.extend(
            float(record["proximal_lambda"])
            for record in observability_records
            if "proximal_lambda" in record
        )
        lambda_upper = max(observed_lambdas)
        hankel_scale = 1.0 / float(kernel_size[0] * kernel_size[1])
        terms = {"multiplicity_normalized_hankel": lambda_upper * hankel_scale}
        bounds = psd_quadratic_spectral_bounds(ridge, terms)
        condition_number = float(bounds["condition_number_upper_bound"])
        factor = cg_convergence_factor(condition_number)
        metadata: dict[str, object] = {
            "protocol": "solver-psd-spectral-bound-v1",
            "scope": "free-coordinate multiplicity-normalized Hankel CG operator",
            "available": True,
            "assumptions": [
                "learned Hankel operator is Hermitian PSD with spectral norm at most one",
                "multiplicity-normalized Hankel lifting has spectral norm at most one",
                "normal-operator scale is 1/(kernel_height*kernel_width)",
                "ridge is strictly positive",
            ],
            "ridge": ridge,
            "observed_proximal_lambda_upper_bound": lambda_upper,
            "normal_operator_scale": hankel_scale,
            "term_upper_bounds": terms,
            **bounds,
            "cg_convergence_factor": factor,
            "cg_energy_error_multiplier_training": 2.0 * factor ** int(training_steps),
            "cg_energy_error_multiplier_inference": 2.0 * factor ** int(inference_steps),
        }
        return metadata

    def solve_view_prediction(
        measured_value: torch.Tensor,
        dc_mask: torch.Tensor,
        projector: torch.Tensor,
        *,
        proximal_lambda: float,
        steps: int,
    ) -> torch.Tensor:
        return solve_hankel_nullspace_proximal_torch(
            measured_value,
            dc_mask,
            projector,
            (3, 3),
            proximal_lambda=proximal_lambda,
            ridge=0.001,
            steps=steps,
        )

    for step in range(1, int(args.steps) + 1):
        model.train()
        input1, target1, probability1 = draw_tensor(2 * (step - 1))
        input2, target2, probability2 = draw_tensor(2 * (step - 1) + 1)
        projector1 = predict_projector(measured * input1[:, None], input1)
        projector2 = predict_projector(measured * input2[:, None], input2)
        prediction1 = solve_view_prediction(
            measured,
            input1,
            projector1,
            proximal_lambda=proximal_lambda_for(input1, label="train_view1"),
            steps=training_solver_steps,
        )
        prediction2 = solve_view_prediction(
            measured,
            input2,
            projector2,
            proximal_lambda=proximal_lambda_for(input2, label="train_view2"),
            steps=training_solver_steps,
        )
        loss1 = physical_ipw_heldout_sum(prediction1, measured, target1, probability1)
        loss2 = physical_ipw_heldout_sum(prediction2, measured, target2, probability2)
        normalizer = float(2 * measured.shape[1] * max(int(target1.sum()), 1))
        measurement_loss = (loss1 + loss2) / normalizer
        subspace_loss = subspace_consistency_loss_torch(projector1, projector2)
        loss = measurement_loss + 0.1 * subspace_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append(
            {
                "step": step,
                "loss": _scalar(loss),
                "measurement_loss": _scalar(measurement_loss),
                "subspace_loss": _scalar(subspace_loss),
                "projector_trace": _scalar(
                    torch.real(torch.diagonal(projector1, dim1=-2, dim2=-1).sum(-1)).mean()
                ),
                "nullspace_energy": _scalar(hankel_nullspace_energy_torch(prediction1, projector1, (3, 3))),
            }
        )

        if step % 25 != 0 and step != int(args.steps):
            continue
        model.eval()
        with torch.no_grad():
            validation_target = torch.as_tensor(split.validation_mask, dtype=torch.bool, device=device)
            validation_input = torch.as_tensor(
                split.acquired_mask & ~split.validation_mask,
                dtype=torch.bool,
                device=device,
            )[None]
            validation_projector = predict_projector(
                measured * validation_input[:, None], validation_input
            )
            validation_prediction = solve_view_prediction(
                measured,
                validation_input,
                validation_projector,
                proximal_lambda=proximal_lambda_for(validation_input, label="validation"),
                steps=24,
            )
            validation_denominator = float(measured.shape[1] * max(int(validation_target.sum()), 1))
            validation_loss_tensor = physical_ipw_heldout_sum(
                validation_prediction,
                measured,
                validation_target,
                torch.ones_like(validation_target, dtype=torch.float32),
            ) / validation_denominator
            fold_losses = []
            for fold_target_np in validation_fold_masks_np:
                fold_target = torch.as_tensor(fold_target_np, dtype=torch.bool, device=device)
                fold_losses.append(
                    physical_ipw_heldout_sum(
                        validation_prediction,
                        measured,
                        fold_target,
                        torch.ones_like(fold_target, dtype=torch.float32),
                    )
                    / float(measured.shape[1] * max(int(fold_target.sum()), 1))
                )
            validation_selection_loss = torch.stack(fold_losses).max()
            validation_support = validation_target[None, None].expand_as(measured)
            validation_error = (validation_prediction - measured).abs().square()[validation_support].sum()
            validation_reference_energy = measured.abs().square()[validation_support].sum().clamp_min(1e-12)
            validation_relative_l2 = torch.sqrt(validation_error / validation_reference_energy)
        validation_value = _scalar(validation_loss_tensor)
        validation_history.append(
            {
                "step": step,
                "gamma_loss": validation_value,
                "gamma_selection_loss": _scalar(validation_selection_loss),
                "gamma_fold_losses": [_scalar(value) for value in fold_losses],
                "gamma_relative_l2": _scalar(validation_relative_l2),
            }
        )
        selection_value = _scalar(validation_selection_loss)
        if selection_value < best_validation - 1e-5:
            best_validation = selection_value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_intervals = 0
        else:
            bad_intervals += 1
            if bad_intervals >= 6:
                break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    inference_steps = 24
    with torch.no_grad():
        full_projector = predict_projector(measured, acquired)
        zero_init_reconstruction = solve_view_prediction(
            measured,
            acquired,
            full_projector,
            proximal_lambda=proximal_lambda_for(acquired, label="inference_zero_init"),
            steps=inference_steps,
        )
    zero_init_np = zero_init_reconstruction[0].detach().cpu().numpy()
    reconstruction_np = zero_init_np

    sake_kernel_size = (6, 6)
    sake_np = sake_kspace_completion(
        measured[0],
        acquired[0],
        kernel_size=sake_kernel_size,
        rank_factor=1.5,
        iterations=100,
        verbose=False,
    ).detach().cpu().numpy()

    output_root.mkdir(parents=True, exist_ok=True)
    reconstruction_arrays = {
        "kspace": reconstruction_np,
        "kspace_zero_init": zero_init_np,
        "sake": sake_np,
        "input": np.asarray(str(input_path)),
        "sampling_kind": np.asarray(str(sample.sampling_kind)),
        "requested_acceleration": np.asarray(
            -1 if sample.requested_acceleration is None else int(sample.requested_acceleration)
        ),
        "crop_shape": np.asarray(cropped_mask.shape, dtype=np.int64),
        "mask": np.asarray(mask_full, dtype=bool),
    }
    np.savez_compressed(output_root / "reconstruction.npz", **reconstruction_arrays)
    scores = _score_post_lock(
        input_path,
        reconstruction_np,
        sake_np,
        crop_bounds,
        normalization_percentile=99.0,
    )

    metadata = {
        "protocol": "hssrecon-demo-v1",
        "input": str(input_path),
        "sampling_kind": sample.sampling_kind,
        "requested_acceleration": sample.requested_acceleration,
        "crop_bounds": list(crop_bounds),
        "crop_shape": list(cropped_mask.shape),
        "coils": int(measured.shape[1]),
        "hankel": {
            "geometry": "full_2d",
            "geometry_semantics": "local_two_dimensional_hankel_windows",
            "kernel_size": list(kernel_size),
            "filter_rank": filter_rank,
            "sampling_detail_rank": 8,
            "sampling_detail_weight": 0.1,
        },
        "self_supervision": {
            "physical_group_split_before_hankel": True,
            "hankel_multiplicity_normalization": True,
            "target_fraction": 0.25,
            "validation_input_mode": "deployment_consistent",
            "validation_fold_count_requested": 2,
            "validation_fold_count_used": int(len(validation_group_folds)),
            "validation_aggregation": "max",
            "reference_free_training_and_selection": True,
            "reference_used_only_for_post_lock_scoring": True,
            "selection_loss": "physical_ipw_heldout_only",
            "selection_auxiliary_weights": {},
            "contract": {
                **contract_summary(
                    strict_receptive_support_audit=bool(
                        contract_audits and contract_audits[0].get("strict_claim_ready", False)
                    )
                ),
                "audit": contract_audits[0] if contract_audits else None,
            },
        },
        "initialization": {"mode": "zero_fill", "sake_used_for_initialization": False},
        "solver": {
            "mode": "fixed_cg",
            "training_steps": training_solver_steps,
            "inference_steps": inference_steps,
            "lambda": 1.0,
            "ridge": 0.001,
            "hard_acquired_data_consistency": True,
        },
        "baselines": {
            "sake": {
                "comparison_only": True,
                "kernel_size": list(sake_kernel_size),
                "iterations": 100,
                "rank_factor": 1.5,
            }
        },
        "training": {
            "steps_requested": int(args.steps),
            "steps_completed": len(history),
            "best_validation_loss": float(best_validation),
            "stopping_rule": "reference_free_Gamma_validation_early_stopping",
            "validation_group_folds": [[int(value) for value in group_ids] for group_ids in validation_group_folds],
        },
        "theory_spectral_bound": theory_spectral_bound_metadata(
            training_solver_steps, inference_steps
        ),
        "split": split.metadata(hankel_kernel_size=(3, 3)),
        "scores": scores,
        "runtime_seconds": float(time.time() - start_time),
    }
    (output_root / "config_and_scores.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "training_history.json").write_text(
        json.dumps({"training": history, "validation": validation_history}, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(best_state, output_root / "checkpoint_best.pt")
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if int(args.steps) <= 0:
        raise ValueError("steps must be positive")
    for input_path in args.inputs:
        case_name = input_path.stem
        metadata = _run_case(args, input_path.expanduser().resolve(), args.output_root / case_name)
        print(json.dumps({"case": case_name, "scores": metadata["scores"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
