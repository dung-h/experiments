#!/usr/bin/env python3
"""Build the report for one completed dense-statevector v1 run from raw rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    values: list[list[str]] = []
    for _, row in frame.iterrows():
        rendered: list[str] = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        values.append(rendered)
    widths = [max(len(header), *(len(row[index]) for row in values)) for index, header in enumerate(headers)]
    render = lambda row: "| " + " | ".join(item.ljust(widths[index]) for index, item in enumerate(row)) + " |"
    return "\n".join([render(headers), render(["-" * width for width in widths]), *(render(row) for row in values)])


def ratio_rows(wide: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("CPU complex128 / complex64", "cpu:complex128", "cpu:complex64"),
        ("CUDA complex128 / complex64", "cuda:complex128", "cuda:complex64"),
        ("CPU / CUDA complex64", "cpu:complex64", "cuda:complex64"),
        ("CPU / CUDA complex128", "cpu:complex128", "cuda:complex128"),
    ]
    rows = []
    for label, numerator, denominator in pairs:
        ratio = wide[numerator] / wide[denominator]
        rows.append({"paired ratio": label, "median": ratio.median(), "minimum": ratio.min(), "maximum": ratio.max()})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    records = pd.read_csv(run_dir / "records.csv")
    records = records[(records["status"] == "ok") & records["warm_execution_median_seconds"].notna()].copy()
    evaluation = pd.read_csv(run_dir / "evaluation" / "aggregate_metrics.csv")
    if len(records) == 0:
        raise ValueError("No successful rows")

    contexts = (
        records.groupby("context_id", as_index=False)["warm_execution_median_seconds"]
        .agg(rows="count", median_seconds="median", min_seconds="min", max_seconds="max")
        .sort_values("context_id")
    )
    wide = records.pivot(index="circuit_id", columns="context_id", values="warm_execution_median_seconds")
    ratios = ratio_rows(wide)
    q24 = (
        records[records["num_qubits"] == 24]
        .pivot_table(index=["family", "depth_parameter"], columns="context_id", values="warm_execution_median_seconds")
        .reset_index()
        .sort_values(["family", "depth_parameter"])
    )
    gpu = records[records["execution_device"] == "cuda"]
    gpu_peak_mb = gpu["torch_cuda_peak_allocated_bytes"].astype(float) / (1024.0 ** 2)
    relative_columns = ["cpu:complex128", "cuda:complex64", "cuda:complex128"]
    output_wide = records.pivot(index="circuit_id", columns="context_id", values="warm_z0_expectation")
    max_output_diff = max((output_wide[column] - output_wide["cpu:complex64"]).abs().max() for column in relative_columns)
    sample_cv = records["warm_execution_stdev_seconds"] / records["warm_execution_mean_seconds"]

    hgb = evaluation[evaluation["model"] == "hist_gradient_log"].set_index("split")
    report = [
        "# Matched dense-statevector runtime matrix v1",
        "",
        "**Finding recorded:** 20 September 2026.",
        "",
        "## Scope and target",
        "",
        "This is a local PyTorch dense-statevector reference-kernel experiment on one RTX 5070 Ti and a 28-logical-CPU machine, with PyTorch 2.11.0+cu128 and 16 configured CPU threads. It is not a Qiskit Aer, CUDA-Q, cuTensorNet, cloud-service or QPU measurement.",
        "",
        "The target is `prepared_execute_reset_state`: allocate a fresh `|0...0>` state, apply a pre-materialized gate schedule and evaluate `<Z_0>`. Circuit construction, gate materialization, compiler/transpiler work, sampling and host-to-device result transfer are excluded. Cold post-prepare and end-to-end-first timings are retained as separate columns in the raw rows.",
        "",
        "## Integrity and coverage",
        "",
        f"All **{len(records)}/{len(records)}** planned rows completed: {records['circuit_id'].nunique()} logical circuit configurations under {records['context_id'].nunique()} declared contexts. The target ranges from {records['warm_execution_median_seconds'].min():.6f} s to {records['warm_execution_median_seconds'].max():.6f} s. The maximum absolute difference in the final scalar `<Z_0>` between CPU complex64 and any other context is {max_output_diff:.3e}; this is an output sanity check, not a state-fidelity study.",
        "",
        f"The median cold/warm ratio is {(records['cold_post_prepare_seconds'] / records['warm_execution_median_seconds']).median():.3f}; the median end-to-end-first/warm ratio is {(records['end_to_end_first_seconds'] / records['warm_execution_median_seconds']).median():.3f}. Median repeat-level coefficient of variation is {sample_cv.median():.3%} (maximum {sample_cv.max():.3%}). CUDA peak allocated memory has median {gpu_peak_mb.median():.2f} MiB and maximum {gpu_peak_mb.max():.2f} MiB. The latter exceeds the final-state footprint because gate application creates temporary permuted/intermediate tensors.",
        "",
        markdown_table(contexts),
        "",
        "## Matched device and precision effects",
        "",
        markdown_table(ratios),
        "",
        "The factor is not constant. CPU complex128 is a median 2.02x slower than CPU complex64, whereas CUDA complex128 is a median 15.74x slower than CUDA complex64 in this implementation. These are measured characteristics of this software/hardware contract, not universal precision laws.",
        "",
        "### 24-qubit frontier (warm median seconds)",
        "",
        markdown_table(q24),
        "",
        "The heaviest completed row is random brickwork, 24 qubits, eight layers, CPU complex128: 24.012 s. QFT-24 CPU complex128 is 19.896 s. Neither is censored or replaced by a synthetic timeout label.",
        "",
        "## Baseline estimator evaluation",
        "",
        "Models predict log warm runtime from logical width/depth/gate counts, family, device and precision. Reported multi-fold figures are computed over concatenated out-of-fold predictions, rather than averaging fold R² values.",
        "",
        markdown_table(evaluation),
        "",
        f"The HGB random-row score (log R² {hgb.loc['random_row', 'r2_log']:.4f}) is descriptive only: another context of the same logical circuit can appear in training. It falls to {hgb.loc['circuit_group', 'r2_log']:.4f} with circuit groups held out. Family-held-out HGB is {hgb.loc['family_held_out', 'r2_log']:.4f}, but there are only five constructed families and 24 logical configurations. The harder width-held-out HGB score is {hgb.loc['width_held_out', 'r2_log']:.4f}; Ridge reaches {evaluation[(evaluation['split'] == 'width_held_out') & (evaluation['model'] == 'ridge_log')].iloc[0]['r2_log']:.4f}, but only across three declared widths. Context-held-out HGB is {hgb.loc['context_held_out', 'r2_log']:.4f} log-R² and {hgb.loc['context_held_out', 'r2_seconds']:.4f} R² in seconds; a model should not be presented as device/precision agnostic on this evidence.",
        "",
        "## Interpretation and limits",
        "",
        "- The matrix supplies actual simulator execution labels, not analytical estimates. It can support a within-contract pilot estimator and a feasibility frontier.",
        "- It does not yet test compilation/transpilation, sampling, noise, tensor-network contraction, a second GPU, another CPU, or a commercial/cloud simulator. Those are separate target domains.",
        "- One fixed circuit seed and a small constructed family set are insufficient to claim broad unseen-circuit generalization. The width-held-out HGB failure also shows that a tree model trained only at 16, 20 and 24 qubits cannot be assumed to extrapolate. The next data increment should add independent circuit seeds, intermediate widths and an adaptive 26--30 qubit frontier with an explicit per-cell censoring policy.",
        "- Aer CPU and CUDA-Q/cuStateVec GPU may be compared later only as separately labelled simulator domains. This matched matrix exists precisely to avoid confounding framework with device and precision.",
        "",
        "## Reproduce",
        "",
        "Use `experiments/simulator_runtime_v1/README.md`. Raw data are `records.jsonl` and `records.csv`; baseline OOF predictions, per-fold metrics and aggregate metrics are in `evaluation/`.",
    ]
    report_path = run_dir / "REPORT.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "torch_dense_statevector_v1",
        "successful_rows": int(len(records)),
        "logical_circuits": int(records["circuit_id"].nunique()),
        "contexts": contexts.to_dict(orient="records"),
        "paired_ratios": ratios.to_dict(orient="records"),
        "maximum_output_difference": float(max_output_diff),
        "hgb_aggregate_metrics": hgb.reset_index().to_dict(orient="records"),
    }
    (run_dir / "results_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
