#!/usr/bin/env python3
"""Build two predictive-runtime figures from the frozen E6 aggregate.

This script reads only the published predictive_runtime_aggregate_v1 CSVs and
writes only presentations/figures_v2/. It never reads raw experiment logs or
starts/repeats a benchmark.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1"
OUT = ROOT / "presentations/figures_v2"

INK = "#172B4D"
MUTED = "#52677F"
GRID = "#E6EBF1"
BLUE = "#2563EB"
TEAL = "#0F766E"
ORANGE = "#D97706"
PURPLE = "#7C3AED"

INPUTS = {
    "aggregate_manifest.json": "aggregate_manifest.json",
    "source_hashes.json": "source_hashes.json",
    "aer_azizov_hash_metrics.csv": "aer_azizov_hash_metrics.csv",
    "aer_azizov_paired_bootstrap.csv": "aer_azizov_paired_bootstrap.csv",
    "mps_fixed_chi16_hash_metrics.csv": "mps_fixed_chi16_hash_metrics.csv",
    "mps_fixed_chi16_paired_bootstrap.csv": "mps_fixed_chi16_paired_bootstrap.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(name: str) -> list[dict[str, str]]:
    with (AGG / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_inputs() -> dict[str, str]:
    manifest_path = AGG / "aggregate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual: dict[str, str] = {}
    for name in INPUTS:
        path = AGG / name
        actual[name] = sha256(path)
        if name in manifest.get("files", {}):
            pinned = manifest["files"][name]["sha256"]
            if actual[name] != pinned:
                raise ValueError(f"aggregate manifest hash mismatch for {name}")
    if manifest.get("artifact_id") != "predictive-runtime-aggregate-v1":
        raise ValueError("unexpected predictive aggregate artifact_id")
    return actual


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def paired_row(rows: list[dict[str, str]], *, domain: str, population: str,
               candidate: str, reference: str) -> dict[str, str]:
    matches = [r for r in rows if r["domain_id"] == domain
               and r["population"] == population
               and r["candidate_method_id"] == candidate
               and r["reference_method_id"] == reference
               and r["status"] == "evaluated"]
    if len(matches) != 1:
        raise ValueError(f"expected one paired row, got {len(matches)}: {domain}, {population}, {candidate}, {reference}")
    row = matches[0]
    if int(row["bootstrap_replicates_completed"]) != 10000:
        raise ValueError("paired bootstrap is incomplete")
    if row["coverage_asymmetric"].lower() != "false":
        raise ValueError("refusing a paired contrast with asymmetric coverage")
    return row


def metric_row(rows: list[dict[str, str]], *, domain: str, population: str,
               method: str) -> dict[str, str]:
    matches = [r for r in rows if r["domain_id"] == domain and r["population"] == population
               and r["method_id"] == method and r["status"] == "evaluated"]
    if len(matches) != 1:
        raise ValueError(f"expected one metric row, got {len(matches)}: {domain}, {population}, {method}")
    return matches[0]


def paired_plot_data(row: dict[str, str]) -> tuple[float, float, float, int]:
    point = float(row["observed_mae_difference_seconds"])
    lo = float(row["bootstrap_ci_low_seconds"])
    hi = float(row["bootstrap_ci_high_seconds"])
    n = int(row["shared_hashes"])
    if not lo <= point <= hi or n < 1:
        raise ValueError("invalid paired estimate / interval / denominator")
    return point, lo, hi, n


def export(fig, stem: str, description: str, pins: dict[str, str]) -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    png, svg = OUT / f"{stem}.png", OUT / f"{stem}.svg"
    # Full content hashes live in the adjacent README; rendering every hash in
    # the plot footer crowds the scientific note and hurts legibility.
    fig.text(0.055, 0.025, "Data: predictive-runtime-aggregate-v1 · source hashes in README.md",
             ha="left", va="bottom", color=MUTED, fontsize=7.0)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white",
                metadata={"Title": stem.replace("_", " "), "Description": description})
    fig.savefig(svg, bbox_inches="tight", facecolor="white",
                metadata={"Title": stem.replace("_", " "), "Description": description,
                          "Creator": "plot_predictive_figures_v2.py", "Date": "2026-10-02"})
    plt.close(fig)
    return {"png": sha256(png), "svg": sha256(svg)}


def build_aer(metrics: list[dict[str, str]], boot: list[dict[str, str]], pins: dict[str, str]) -> dict:
    domain = "aer_azizov_core_q9"
    # Use paired comparisons from the frozen aggregate; positive means the
    # candidate listed in the comparison has higher MAE than its reference.
    specs = [
        ("Source GNN − transpiled GNN", "azizov_gnn_source_median_three_seeds",
         "azizov_gnn_transpiled_median_three_seeds", BLUE),
        ("Hybrid GNN − transpiled GNN", "azizov_gnn_hybrid_median_three_seeds",
         "azizov_gnn_transpiled_median_three_seeds", TEAL),
        ("Transpiled GNN − XGBoost", "azizov_gnn_transpiled_median_three_seeds",
         "classical_xgboost_transpiled", ORANGE),
    ]
    rows = [paired_row(boot, domain=domain, population="all_finite_targets", candidate=c, reference=r)
            for _, c, r, _ in specs]
    observations = [paired_plot_data(r) for r in rows]
    expected_set_hash = {r["shared_hash_set_sha256"] for r in rows}
    if len(expected_set_hash) != 1 or {n for _, _, _, n in observations} != {150}:
        raise ValueError("Aer contrasts do not use the same 150-hash paired set")

    trans = metric_row(metrics, domain=domain, population="all_finite_targets",
                       method="azizov_gnn_transpiled_median_three_seeds")
    hybrid = metric_row(metrics, domain=domain, population="all_finite_targets",
                        method="azizov_gnn_hybrid_median_three_seeds")
    source = metric_row(metrics, domain=domain, population="all_finite_targets",
                        method="azizov_gnn_source_median_three_seeds")
    xgb = metric_row(metrics, domain=domain, population="all_finite_targets",
                     method="classical_xgboost_transpiled")
    if any(int(r["scored_hashes"]) != 150 for r in (trans, hybrid, source, xgb)):
        raise ValueError("Aer metric rows are not scored on all 150 assigned rows")

    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    fig.suptitle("Aer warm runtime: paired error contrasts", x=0.06, y=0.97,
                 ha="left", color=INK, fontsize=17, fontweight="bold")
    fig.text(0.06, 0.918,
             "One local Aer target · 150/150 shared circuit hashes · ΔMAE = candidate − reference (seconds)",
             ha="left", va="top", color=MUTED, fontsize=9.5)
    yy = list(range(len(specs)))[::-1]
    for y, ((label, _, _, color), (point, lo, hi, n)) in zip(yy, zip(specs, observations)):
        ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", color=color,
                    ecolor=color, elinewidth=2.1, capsize=4, markersize=7, zorder=3)
        ax.text(hi + 0.012, y, f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]  n={n}",
                va="center", fontsize=8.2, color=INK)
    ax.axvline(0, color=INK, linewidth=1, linestyle="--", alpha=0.8)
    ax.set_yticks(yy, [s[0] for s in specs])
    ax.set_xlabel("Paired difference in MAE (seconds); 95% grouped-bootstrap interval", color=MUTED)
    ax.set_xlim(-0.23, 0.47)
    ax.set_ylim(-0.7, 2.8)
    ax.tick_params(axis="y", length=0)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    metrics_line = (
        f"MAE: source {float(source['mae_seconds']):.3f}s · hybrid {float(hybrid['mae_seconds']):.3f}s · "
        f"transpiled {float(trans['mae_seconds']):.3f}s · XGBoost(transpiled) {float(xgb['mae_seconds']):.3f}s. "
        "Source-vs-transpiled interval excludes zero; hybrid-vs-transpiled and GNN-vs-XGBoost intervals include zero."
    )
    fig.text(0.06, 0.095, metrics_line, color=MUTED, fontsize=8.2, wrap=True)
    fig.text(0.06, 0.065,
             "Intervals are unadjusted 95% percentile intervals; they do not establish universal superiority or transfer to other Aer targets.",
             color=MUTED, fontsize=8.0, wrap=True)
    fig.tight_layout(rect=[0.045, 0.17, 0.98, 0.88])
    description = (
        "Paired MAE differences on the same 150 source_sha256 groups for a single local Aer q9 warm-execution target. "
        "Positive delta means the named candidate has worse MAE. Intervals are the 10,000-replicate 95% grouped "
        "percentile bootstrap intervals from the aggregate and are unadjusted for multiple comparisons."
    )
    file_hashes = export(fig, "fig1_aer_paired_predictor_contrasts", description, pins)
    return {
        "file_hashes": file_hashes,
        "row_set": "150 shared circuit hashes; source_sha256 grouped bootstrap",
        "evaluation_target_clock": "aer_azizov_core_q9/warm_execution",
        "method_output_clock": "predicted_noisy_aer_warm_execution_seconds",
        "caption": description,
        "contrasts": [{"label": s[0], "delta_mae_seconds": o[0], "ci95_low": o[1],
                       "ci95_high": o[2], "paired_n": o[3]} for s, o in zip(specs, observations)],
        "metric_context": {"source_gnn_mae": float(source["mae_seconds"]),
                           "hybrid_gnn_mae": float(hybrid["mae_seconds"]),
                           "transpiled_gnn_mae": float(trans["mae_seconds"]),
                           "transpiled_xgboost_mae": float(xgb["mae_seconds"])},
    }


def build_mps(metrics: list[dict[str, str]], boot: list[dict[str, str]], pins: dict[str, str]) -> dict:
    domain = "mps_fixed_chi16"
    specs = [
        ("Family residual − family agnostic", "e5_family_residual_median_three_seeds",
         "e5_family_agnostic_median_three_seeds", PURPLE),
        ("Family residual − E3 graph", "e5_family_residual_median_three_seeds",
         "e3_fixed_mps_graph_median_three_seeds", BLUE),
        ("Family residual − E3 Ridge", "e5_family_residual_median_three_seeds",
         "e3_fixed_mps_ridge_alpha_1", TEAL),
    ]
    populations = ["all_finite_targets", "quality_pass_only"]
    panel_data: dict[str, list[tuple[float, float, float, int]]] = {}
    for population in populations:
        rows = [paired_row(boot, domain=domain, population=population, candidate=c, reference=r)
                for _, c, r, _ in specs]
        observed = [paired_plot_data(r) for r in rows]
        if len({r["shared_hash_set_sha256"] for r in rows}) != 1:
            raise ValueError(f"MPS contrasts use different paired hashes within {population}")
        expected_n = 144 if population == "all_finite_targets" else 142
        if {n for _, _, _, n in observed} != {expected_n}:
            raise ValueError(f"MPS {population} denominator is not {expected_n}")
        panel_data[population] = observed

    all_metric_rows = [metric_row(metrics, domain=domain, population="all_finite_targets", method=m)
                       for m in ("e3_fixed_mps_graph_median_three_seeds",
                                 "e5_family_residual_median_three_seeds",
                                 "e5_family_agnostic_median_three_seeds",
                                 "e3_fixed_mps_ridge_alpha_1")]
    quality_rows = [metric_row(metrics, domain=domain, population="quality_pass_only", method=m)
                    for m in ("e3_fixed_mps_graph_median_three_seeds",
                              "e5_family_residual_median_three_seeds",
                              "e5_family_agnostic_median_three_seeds",
                              "e3_fixed_mps_ridge_alpha_1")]
    if any(int(r["scored_hashes"]) != n for r, n in
           [(all_metric_rows[0], 144), (all_metric_rows[1], 144), (quality_rows[0], 142), (quality_rows[1], 142)]):
        raise ValueError("MPS model metric coverage differs from plotted denominator")

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.1), sharex=True, sharey=True)
    fig.suptitle("Fixed-χ16 CUDA-Q MPS: runtime-model contrasts", x=0.06, y=0.97,
                 ha="left", color=INK, fontsize=17, fontweight="bold")
    fig.text(0.06, 0.918,
             "FP64, bond dimension 16 · evaluation target is MPS warm state execution · ΔMAE = candidate − reference (seconds)",
             ha="left", va="top", color=MUTED, fontsize=9.2)
    for ax, population, heading in zip(axes, populations, ["All finite targets (n=144)", "Quality-pass only (n=142)"]):
        yy = list(range(len(specs)))[::-1]
        for y, (spec, obs) in enumerate(zip(specs, panel_data[population])):
            label, _, _, color = spec
            pos = yy[y]
            point, lo, hi, n = obs
            ax.errorbar(point, pos, xerr=[[point - lo], [hi - point]], fmt="o", color=color,
                        ecolor=color, elinewidth=2.0, capsize=4, markersize=6.5, zorder=3)
            ax.text(hi + 0.04, pos, f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]", 
                    va="center", fontsize=7.6, color=INK)
        ax.axvline(0, color=INK, linewidth=1, linestyle="--", alpha=0.8)
        ax.set_title(heading, loc="left", color=INK, fontsize=10.5, fontweight="bold")
        ax.set_yticks(yy, [s[0] for s in specs])
        ax.set_xlim(-0.62, 1.78)
        ax.set_ylim(-0.7, 2.8)
        ax.tick_params(axis="y", length=0, labelsize=8)
        style_axes(ax)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("Paired difference in MAE (seconds); 95% grouped-bootstrap interval", color=MUTED)
    axes[1].set_xlabel("Same target clock and units", color=MUTED)
    graph, residual, agnostic, ridge = all_metric_rows
    metric_context = (
        f"All-finite MAE: E3 graph {float(graph['mae_seconds']):.3f}s · E5 family residual "
        f"{float(residual['mae_seconds']):.3f}s · E5 family agnostic {float(agnostic['mae_seconds']):.3f}s · "
        f"E3 Ridge {float(ridge['mae_seconds']):.3f}s. Family residual vs family-agnostic interval crosses zero; "
        "family-residual vs E3 graph is positive (worse)."
    )
    fig.text(0.06, 0.095, metric_context, color=MUTED, fontsize=8.0, wrap=True)
    fig.text(0.06, 0.060,
             "The MPS quality gate is fidelity ≥0.99. All-finite includes two finite but quality-failed circuits; quality-pass is a prespecified subset. Intervals are unadjusted.",
             color=MUTED, fontsize=7.8, wrap=True)
    fig.tight_layout(rect=[0.04, 0.18, 0.985, 0.88], w_pad=2.6)
    description = (
        "Paired MAE contrasts on a fixed CUDA-Q MPS FP64 bond-dimension-16 warm-state-execution target. "
        "The left panel uses 144 finite paired targets; the right uses the prespecified 142 quality-pass subset "
        "(fidelity at least 0.99). Intervals are 10,000-replicate 95% source_sha256-grouped percentile intervals, "
        "unadjusted for multiple comparisons. Positive delta means the family-residual candidate has worse MAE."
    )
    file_hashes = export(fig, "fig2_mps_paired_predictor_contrasts", description, pins)
    return {
        "file_hashes": file_hashes,
        "row_set": {"all_finite": "144/144 finite target rows", "quality_pass": "142/142 fidelity-pass rows"},
        "evaluation_target_clock": "cudaq_mps_fp64_bond16_warm_state_execution_seconds",
        "method_output_clock": "predicted_cudaq_mps_fp64_bond16_warm_state_execution_seconds",
        "quality_gate": "fidelity >= 0.99",
        "caption": description,
        "contrasts": {population: [{"label": s[0], "delta_mae_seconds": o[0], "ci95_low": o[1],
                                    "ci95_high": o[2], "paired_n": o[3]}
                                   for s, o in zip(specs, panel_data[population])]
                      for population in populations},
        "all_finite_mae_seconds": {r["method_id"]: float(r["mae_seconds"]) for r in all_metric_rows},
        "quality_pass_mae_seconds": {r["method_id"]: float(r["mae_seconds"]) for r in quality_rows},
    }


def main() -> None:
    pins = verify_inputs()
    aer = build_aer(read_csv("aer_azizov_hash_metrics.csv"),
                    read_csv("aer_azizov_paired_bootstrap.csv"), pins)
    mps = build_mps(read_csv("mps_fixed_chi16_hash_metrics.csv"),
                    read_csv("mps_fixed_chi16_paired_bootstrap.csv"), pins)
    script_path = Path(__file__).resolve()
    lines = [
        "# Simulator predictive-runtime figures (E6 aggregate)",
        "",
        "Two separate figures are provided because Aer and CUDA-Q MPS have different evaluation targets and clocks. Neither chart ranks a method across those targets. They use only the frozen E6 predictive aggregate; no raw timing or retraining was performed.",
        "",
        "## Figure 1 — Aer warm-execution paired contrasts",
        "",
        f"Files: `fig1_aer_paired_predictor_contrasts.png` (SHA-256 `{aer['file_hashes']['png']}`), `fig1_aer_paired_predictor_contrasts.svg` (SHA-256 `{aer['file_hashes']['svg']}`).",
        "",
        f"Population: {aer['row_set']}. Target clock: `{aer['evaluation_target_clock']}`; prediction clock: `{aer['method_output_clock']}`. Each Δ is candidate MAE minus reference MAE on the same source-hash groups; positive means the candidate has larger error. Error bars are source_sha256-grouped 10,000-replicate 95% percentile bootstrap intervals. The source-to-transpiled graph contrast excludes zero; the hybrid-to-transpiled and transpiled-GNN-to-XGBoost intervals include zero. These unadjusted intervals do not establish general superiority or transfer beyond this single local Aer target.",
        "",
        "MAE context on 150/150 rows: source GNN 0.450 s; hybrid GNN 0.406 s; transpiled GNN 0.301 s; transpiled XGBoost 0.297 s.",
        "",
        "## Figure 2 — fixed-χ16 MPS warm-state-execution paired contrasts",
        "",
        f"Files: `fig2_mps_paired_predictor_contrasts.png` (SHA-256 `{mps['file_hashes']['png']}`), `fig2_mps_paired_predictor_contrasts.svg` (SHA-256 `{mps['file_hashes']['svg']}`).",
        "",
        f"Target clock: `{mps['evaluation_target_clock']}`; prediction clock: `{mps['method_output_clock']}`. The left panel has 144 finite paired rows; the right is the prespecified 142-row fidelity-pass subset (`fidelity ≥ 0.99`). Two finite but quality-failed target rows occur only in the all-finite population. Δ is E5 family-residual MAE minus each named reference; positive means E5 error is larger. Intervals are unadjusted source_sha256-grouped 10,000-replicate 95% percentile bootstrap intervals. Family-residual vs family-agnostic includes zero in both panels; residual-vs-E3-graph is positive. These are comparisons within the fixed MPS configuration, not with Aer.",
        "",
        "All-finite MAE context: E3 graph 0.302 s; E5 family residual 1.001 s; E5 family agnostic 1.081 s; E3 Ridge 1.229 s.",
        "",
        "## Source pins",
        "",
        "The aggregate manifest validates each plotted CSV hash. The source hash registry also pins the underlying E6 data/code/environment inputs.",
    ]
    for name, digest in pins.items():
        lines.append(f"- `artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/{name}` — SHA-256 `{digest}`")
    lines += [
        f"- `{script_path.relative_to(ROOT)}` — SHA-256 `{sha256(script_path)}`",
        "",
        "## Reproduction",
        "",
        "From repository root, with the packages in `presentations/requirements-figures.txt` installed:",
        "",
        "```bash",
        "python presentations/source/plot_predictive_figures_v2.py",
        "```",
        "",
        "The script reads the frozen aggregate, checks its manifest pins, and writes only this directory. It does not modify benchmark data, scorecards, or `CURRENT.json`.",
        "",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"PASS wrote figures to {OUT.relative_to(ROOT)}")
    print(f"Aer: {aer['file_hashes']}")
    print(f"MPS: {mps['file_hashes']}")
    print(f"README SHA-256: {sha256(OUT / 'README.md')}")


if __name__ == "__main__":
    main()
