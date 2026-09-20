#!/usr/bin/env python3
"""Derive audit-friendly explanations from the dated simulator validation pass.

This script does not retrain any model. It aggregates committed OOF decisions,
raw timing rows and feasibility logs into a separate interpretation artifact.
All explanatory statements are framed as evidence-consistent mechanisms rather
than causal proof.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "simulator_papers"


def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    )


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): scalar(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scalar(item) for item in value]
    return value


def records(frame: pd.DataFrame, by: list[str], aggregations: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
    return [scalar(row) for row in frame.groupby(by, dropna=False).agg(**aggregations).reset_index().to_dict("records")]


def maximum(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return max(rows, key=lambda row: float(row[field]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ARTIFACTS / "DEEP_ANALYSIS_2026-09-20",
    )
    args = parser.parse_args()

    # VQCSim: paired base/mirror stability and primary OOF residual structure.
    base = pd.read_csv(ARTIFACTS / "vqcsim_rq2_gpu_fp32_16_24_20260920/records.csv")
    mirror = pd.read_csv(ARTIFACTS / "vqcsim_rq2_gpu_fp32_mirror_16_24_20260920/records.csv")
    paired = base[["family", "num_qubits", "mean_wall_ms"]].merge(
        mirror[["family", "num_qubits", "mean_wall_ms"]],
        on=["family", "num_qubits"], suffixes=("_base", "_mirror"), validate="one_to_one",
    )
    paired["timing_ratio"] = paired["mean_wall_ms_mirror"] / paired["mean_wall_ms_base"]
    paired["absolute_percent_change"] = 100 * (paired["timing_ratio"] - 1).abs()
    vq_oof = pd.read_csv(ARTIFACTS / "vqcsim_estimator_base_mirror_20260920/evaluation_oof_predictions.csv")
    vq_primary = vq_oof[
        (vq_oof["split"] == "circuit_group_5fold")
        & (vq_oof["model"] == "hist_gradient_boosting")
    ].copy()
    vq_primary["absolute_error_ms"] = (vq_primary["actual_ms"] - vq_primary["predicted_ms"]).abs()
    vq_primary["absolute_percent_error"] = 100 * vq_primary["absolute_error_ms"] / vq_primary["actual_ms"]
    vq_by_family = records(vq_primary, ["family"], {
        "n": ("actual_ms", "size"), "mae_ms": ("absolute_error_ms", "mean"),
        "mape_percent": ("absolute_percent_error", "mean"),
    })
    vq_by_width = records(vq_primary, ["num_qubits"], {
        "n": ("actual_ms", "size"), "mae_ms": ("absolute_error_ms", "mean"),
        "mape_percent": ("absolute_percent_error", "mean"),
    })

    # Aer: identify directional backend-shift behavior in OOF interval rows.
    aer_oof = pd.read_csv(ARTIFACTS / "aer_interval_validation_20260920/evaluation_oof_predictions.csv")
    aer_backend = aer_oof[aer_oof["split"] == "backend_held_out"].copy()
    aer_backend["covered"] = (
        (aer_backend["actual_seconds"] >= aer_backend["lower_seconds"])
        & (aer_backend["actual_seconds"] <= aer_backend["upper_seconds"])
    )
    aer_backend["actual_over_predicted"] = aer_backend["actual_seconds"] / aer_backend["predicted_seconds"]
    aer_backend["interval_width_seconds"] = aer_backend["upper_seconds"] - aer_backend["lower_seconds"]
    aer_by_backend = records(aer_backend, ["backend_class"], {
        "n": ("actual_seconds", "size"), "coverage": ("covered", "mean"),
        "median_actual_over_predicted": ("actual_over_predicted", "median"),
        "median_interval_width_seconds": ("interval_width_seconds", "median"),
        "median_log_radius": ("conformal_log_radius", "median"),
    })

    # PPS: controls are pre-run inputs, while retained terms are explicitly excluded.
    pps = load_jsonl(ARTIFACTS / "zero_setup_pps_grid_20260920.jsonl")
    pps_config = records(pps, ["num_trotter_steps", "delta"], {
        "n_trials": ("propagation_seconds", "size"),
        "median_seconds": ("propagation_seconds", "median"),
        "median_num_paulis_post_run": ("num_paulis", "median"),
    })
    pps_delta = records(pps, ["delta"], {
        "median_seconds": ("propagation_seconds", "median"),
        "min_seconds": ("propagation_seconds", "min"),
        "max_seconds": ("propagation_seconds", "max"),
    })
    pps_oof = pd.read_csv(ARTIFACTS / "zero_setup_pps_estimator_20260920/evaluation_oof_predictions.csv")
    pps_delta_oof = pps_oof[
        (pps_oof["split"] == "delta_held_out")
        & (pps_oof["model"] == "hist_gradient_boosting")
    ].copy()
    # Match the estimator's target transform exactly: log(seconds), not log1p.
    pps_delta_oof["absolute_log_error"] = np.abs(
        np.log(pps_delta_oof["actual_seconds"]) - np.log(pps_delta_oof["predicted_seconds"])
    )
    pps_error_by_delta = records(pps_delta_oof, ["delta"], {
        "n": ("actual_seconds", "size"),
        "mean_absolute_log_error": ("absolute_log_error", "mean"),
        "median_actual_seconds": ("actual_seconds", "median"),
        "median_predicted_seconds": ("predicted_seconds", "median"),
    })

    # Candidate selection is conditional on memory-feasible plans.
    plan_root = ARTIFACTS / "explicit_plan_ranking_expanded_20260920"
    plan = pd.read_csv(plan_root / "candidates_connected.csv")
    plan["log_flops"] = np.log10(plan["pre_run_flop_count"])
    plan["log_runtime"] = np.log10(plan["contract_gpu_median_s"])
    plan_correlations = []
    for (family, width), group in plan.groupby(["family", "num_qubits"]):
        plan_correlations.append({
            "family": family,
            "num_qubits": int(width),
            "candidate_count": int(len(group)),
            "spearman_log_flops_log_runtime": float(group["log_flops"].corr(group["log_runtime"], method="spearman")),
        })
    plan_choices = pd.read_csv(plan_root / "evaluation_connected.csv")
    plan_by_family = records(plan_choices, ["family"], {
        "groups": ("flop_selection_regret_ratio", "size"),
        "mean_regret": ("flop_selection_regret_ratio", "mean"),
        "max_regret": ("flop_selection_regret_ratio", "max"),
        "exact_fastest_rate": ("flop_exact_fastest", "mean"),
    })
    plan_failures = json.loads((plan_root / "candidates_connected.failures.json").read_text())
    plan_failure_kinds = {
        "memory_limit": sum("MemoryLimitExceeded" in item["error"] for item in plan_failures),
        "unsupported": sum("NOT_SUPPORTED" in item["error"] for item in plan_failures),
    }

    # EMU-MPS: quantized threshold labels, with no unseen-family evidence.
    emu_base = pd.read_csv(ROOT / "artifacts" / "pasqal_emu_mps" / "threshold_sweep_labels.csv")
    emu_extension = pd.read_csv(ARTIFACTS / "emu_mps_threshold_extension_20260920/threshold_labels.csv")
    emu_labels = pd.concat([emu_base, emu_extension], ignore_index=True)
    emu_labels["required_max_bond_dim"] = emu_labels["required_max_bond_dim"].astype(int)
    emu_by_width = records(emu_labels, ["n_qubits"], {
        "n": ("required_max_bond_dim", "size"),
        "min_threshold": ("required_max_bond_dim", "min"),
        "median_threshold": ("required_max_bond_dim", "median"),
        "max_threshold": ("required_max_bond_dim", "max"),
    })
    emu_oof = pd.read_csv(ARTIFACTS / "family_aware_emu_mps_expanded_20260920/evaluation.csv")
    emu_shared = emu_oof[emu_oof["method"] == "shared_numeric"].copy()
    emu_by_held_width = records(emu_shared, ["held_out_n_qubits"], {
        "n": ("absolute_rung_error", "size"),
        "mean_absolute_rung_error": ("absolute_rung_error", "mean"),
        "exact_rate": ("absolute_rung_error", lambda values: float((values == 0).mean())),
    })

    # Precision labels are one scalar agreement test, not state fidelity.
    precision_root = ARTIFACTS / "precision_selection_expanded_20260920"
    precision = pd.read_csv(precision_root / "policy_canonical_pairs.csv")
    precision["time_ratio_128_over_64"] = precision["complex128_seconds"] / precision["complex64_seconds"]
    precision_by_family = records(precision, ["family"], {
        "n": ("safe_complex64", "size"), "safe_rate": ("safe_complex64", "mean"),
        "median_relative_difference": ("relative_scalar_difference", "median"),
        "max_relative_difference": ("relative_scalar_difference", "max"),
        "median_time_ratio_128_over_64": ("time_ratio_128_over_64", "median"),
    })
    precision_decisions = pd.read_csv(precision_root / "policy_oof_decisions.csv")
    precision_family_decisions = precision_decisions[
        precision_decisions["split"] == "family_held_out"
    ].copy()
    precision_family_decisions["selected_64"] = precision_family_decisions["selected_precision"] == "complex64"
    precision_family_decisions["unsafe_selected_64"] = (
        precision_family_decisions["selected_64"] & ~precision_family_decisions["actual_safe_complex64"]
    )

    payload = {
        "date": "2026-09-20",
        "scope": "Post-hoc descriptive analysis of the dated source-aware simulator validation pass; no model retraining.",
        "vqcsim": {
            "n_matched_base_mirror_pairs": int(len(paired)),
            "median_mirror_over_base_ratio": float(paired["timing_ratio"].median()),
            "mean_absolute_percent_change": float(paired["absolute_percent_change"].mean()),
            "max_absolute_percent_change": float(paired["absolute_percent_change"].max()),
            "primary_oof_by_family": vq_by_family,
            "primary_oof_by_width": vq_by_width,
        },
        "aer": {"backend_held_out_by_backend": aer_by_backend},
        "pps": {
            "median_by_configuration": pps_config,
            "median_by_delta": pps_delta,
            "delta_held_out_hgb_error_by_delta": pps_error_by_delta,
        },
        "plan_ranking": {
            "within_network_spearman": plan_correlations,
            "selection_by_family": plan_by_family,
            "failure_kinds": plan_failure_kinds,
        },
        "emu_mps": {
            "threshold_by_width": emu_by_width,
            "shared_model_error_by_held_width": emu_by_held_width,
        },
        "precision": {
            "agreement_by_family": precision_by_family,
            "family_held_out_complex64_selections": int(precision_family_decisions["selected_64"].sum()),
            "family_held_out_unsafe_complex64_selections": int(precision_family_decisions["unsafe_selected_64"].sum()),
        },
    }
    payload = scalar(payload)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")

    worst_vq_family = maximum(vq_by_family, "mae_ms")
    worst_vq_width = maximum(vq_by_width, "mae_ms")
    worst_pps_delta = maximum(pps_error_by_delta, "mean_absolute_log_error")
    lowest_plan_rho = min(plan_correlations, key=lambda row: row["spearman_log_flops_log_runtime"])
    worst_plan_family = maximum(plan_by_family, "max_regret")
    report = f"""# Deep analysis of the simulator-estimator validation pass

**Date:** 20 September 2026
**Method:** descriptive analysis of existing raw rows and out-of-fold outputs.
No model was retrained for this report.  The mechanisms below are
evidence-consistent explanations, not causal identification.

## VQCSim: high score is credible for this contract, but mainly tests a smooth small generator grid

The base and mirror circuits have a median time ratio of
**{payload['vqcsim']['median_mirror_over_base_ratio']:.3f}x**. Their mean
absolute timing change is **{payload['vqcsim']['mean_absolute_percent_change']:.2f}%**
(maximum **{payload['vqcsim']['max_absolute_percent_change']:.2f}%**).  The
primary circuit-group split therefore did not obtain its score merely by
splitting a base/mirror pair across train and test.

However, the corpus still contains only five generator families and nine
ordered widths.  Its circuit-group HGB residual is largest for
`{worst_vq_family['family']}` (MAE {worst_vq_family['mae_ms']:.3f} ms) and at
width {int(worst_vq_width['num_qubits'])} (MAE {worst_vq_width['mae_ms']:.3f} ms).
The strong grouped score is thus best interpreted as interpolation over smooth
family/size trajectories under fixed hardware and prepared-inference timing.
The lower width-held-out result in the main report is the more relevant warning
for a new width; unseen circuit templates, sampling and CPU paths remain out
of scope.

## Noisy Aer: conformal coverage fails because backend shift violates the exchangeability assumption

| Held-out backend | N | Coverage | Median actual/predicted | Median interval width (s) | Median log radius |
|---|---:|---:|---:|---:|---:|
"""
    for row in aer_by_backend:
        report += (
            f"| {row['backend_class']} | {row['n']} | {100 * row['coverage']:.1f}% | "
            f"{row['median_actual_over_predicted']:.3f}x | {row['median_interval_width_seconds']:.3f} | "
            f"{row['median_log_radius']:.3f} |\n"
        )
    report += """

The conformal residual distribution is estimated from the training backend.
After a fake-backend holdout, compiled gate/routing/noise context changes in a
way that the current feature representation and residual radius do not cover.
The result is not evidence that conformal prediction is inherently invalid;
it is evidence that the current calibration is **not domain-conditional**.
Backend-specific calibration, a richer backend representation, or an explicit
shift detector is required before an interval can be shown to a user.

## PPS: `delta` controls a representation transition rather than a smooth depth effect

Across the controlled grid, the median propagation time by declared delta is:

| Delta | Median seconds | Minimum seconds | Maximum seconds |
|---:|---:|---:|---:|
"""
    for row in pps_delta:
        report += f"| {row['delta']:.0e} | {row['median_seconds']:.4f} | {row['min_seconds']:.4f} | {row['max_seconds']:.4f} |\n"
    report += f"""

The post-run number of retained Pauli terms is deliberately excluded from the
predictor, yet it changes abruptly once truncation becomes less aggressive.
That gives a plausible mechanism for the weak delta-held-out behavior: three
control levels do not identify a stable response curve, and the same logical
circuit can cross a representation-complexity boundary.  The largest HGB
delta-held-out mean absolute log error is at delta `{worst_pps_delta['delta']:.0e}`
({worst_pps_delta['mean_absolute_log_error']:.3f}).  A useful next model needs
more declared control levels and a model class that can express monotone,
piecewise behavior; it must still avoid post-run representation leakage.

## Plan ranking: FLOPs explain much of this feasible set, not all GPU time

The weakest within-network Spearman correlation between log FLOPs and log warm
runtime occurs for `{lowest_plan_rho['family']}` at
{int(lowest_plan_rho['num_qubits'])} qubits
({lowest_plan_rho['spearman_log_flops_log_runtime']:.3f}).  The largest
family-level regret is `{worst_plan_family['family']}`
({worst_plan_family['max_regret']:.3f}x).  Six of seven explicit random plans
are infeasible under the declared 50% VRAM limit; the remaining failure is
unsupported by the cuTensorNet path API.

FLOP count does not encode peak workspace, slicing, memory traffic, kernel
choice, or launch overhead.  More importantly, the 80% selection result is
conditional on the feasible candidate subset.  A deployable ranker must first
model feasibility under a memory budget, then rank among feasible plans; it
cannot learn from a table where infeasible plans are silently removed.

## EMU-MPS: the target is quantized and the current family signal is constructed

| Width | Labels | Minimum threshold | Median threshold | Maximum threshold |
|---:|---:|---:|---:|---:|
"""
    for row in emu_by_width:
        report += (
            f"| {int(row['n_qubits'])} | {row['n']} | {int(row['min_threshold'])} | "
            f"{row['median_threshold']:.1f} | {int(row['max_threshold'])} |\n"
        )
    report += """

Exact rung accuracy is low partly because the label space doubles at each step;
an error of one rung is already a 2x bond-dimension difference.  The 100%
family-classifier result has no generalization meaning here: pulse duration,
segment count, amplitudes and detuning deterministically identify the eight
constructed families.  The residual correction ties the shared model because
the available data are dominated by a monotone width/intensity pattern and
every held-out width still has every family in training.  An unseen-family
claim needs a non-constructed circuit corpus and a family split that removes
the entire family from training.

## Precision: random row splits share numerical regimes that family holdout removes

| Family | Pairs | Safe complex64 rate | Median relative difference | Maximum relative difference | Median 128/64 time |
|---|---:|---:|---:|---:|---:|
"""
    for row in precision_by_family:
        report += (
            f"| {row['family']} | {row['n']} | {100 * row['safe_rate']:.1f}% | "
            f"{row['median_relative_difference']:.3e} | {row['max_relative_difference']:.3e} | "
            f"{row['median_time_ratio_128_over_64']:.3f}x |\n"
        )
    report += f"""

At the declared `5e-7` scalar-relative-difference tolerance, family-held-out
evaluation chooses complex64 only
{payload['precision']['family_held_out_complex64_selections']} times, and
{payload['precision']['family_held_out_unsafe_complex64_selections']} of those
choices are unsafe.  The label is sensitive to circuit family, depth and
contraction path, while the corpus has only two seeds and one scalar output
per pair.  Random folds leak those regimes across train/test; family holdout
exposes their absence.  Moreover, scalar agreement is not state fidelity, so
even a better classifier would need a separate numerical-accuracy validation.

## Engineering implications

1. Keep estimator, feasibility, confidence interval and numerical-precision
   decisions as separate heads with separate labels.
2. Make configuration/domain holdout the release gate. Random-row scores are
   descriptive only.
3. Version backend, precision, memory budget, candidate generator and timing
   inclusion policy with every label.  These are causal runtime inputs, not
   incidental provenance.
4. Collect more independent circuit seeds and configurations before increasing
   model complexity. The current bottleneck is experimental support, not the
   absence of another regressor.
"""
    args.output_prefix.with_suffix(".md").write_text(report)
    print(args.output_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
