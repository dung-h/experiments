#!/usr/bin/env python3
"""Rebuild the four presentation figures from the frozen scorecard tables."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.rcParams["svg.hashsalt"] = "f-c4-frozen-scorecard-figures-v1"


ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3"
PRESENTATION = ROOT / "docs/presentation.md"
OUT = ROOT / "work/presentation-figures"

INK = "#172B4D"
MUTED = "#52677F"
GRID = "#E6EBF1"
BLUE = "#2563EB"
ORANGE = "#D97706"
TEAL = "#0F766E"
PURPLE = "#7C3AED"
RED = "#B42318"
GREEN = "#15803D"
GRAY = "#64748B"


def read_csv(relative: str) -> list[dict[str, str]]:
    with (PACK / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_pin(relative: str) -> dict[str, str]:
    path = PACK / relative
    return {"path": relative, "sha256": sha256(path)}


def configure_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def title_block(fig, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.06, y=0.97, ha="left", color=INK,
                 fontsize=17, fontweight="bold")
    fig.text(0.06, 0.925, subtitle, ha="left", va="top", color=MUTED,
             fontsize=9.5)


def save_pair(fig, stem: str, description: str, sources: list[dict[str, str]]) -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    source_note = " | ".join(f"{item['path']} SHA-256 {item['sha256']}" for item in sources)
    fig.text(0.06, 0.035, source_note, ha="left", va="bottom", color=MUTED,
             fontsize=6.7, wrap=True)
    png = OUT / f"{stem}.png"
    svg = OUT / f"{stem}.svg"
    metadata = {
        "Title": stem.replace("_", " "),
        "Description": description + " Sources: " + source_note,
        "Creator": "F-C4 frozen-scorecard plotting source",
        "Date": "2026-10-02",
    }
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white",
                metadata={"Title": metadata["Title"], "Description": metadata["Description"]})
    fig.savefig(svg, bbox_inches="tight", facecolor="white", metadata=metadata)
    plt.close(fig)
    return {"png": png.name, "png_sha256": sha256(png),
            "svg": svg.name, "svg_sha256": sha256(svg)}


def plot_qpu_paired_mae() -> dict:
    path = "qpu/archived_method_pairwise_comparisons.csv"
    rows = [row for row in read_csv(path)
            if row["candidate_method_id"] == "mali_graph_architecture_v3_large"
            and row["reference_method_id"] == "unified_polynomial_v3"]
    order = ["mali_real_qpu", "qonductor_single_circuit_ibm", "qpack_mcp"]
    labels = {"mali_real_qpu": "Ma–Li source", "qonductor_single_circuit_ibm": "Qonductor source",
              "qpack_mcp": "QPack source"}
    by_source = {row["source_id"]: row for row in rows}
    if set(by_source) != set(order) or len(rows) != 3:
        raise ValueError("paired QPU subset does not match the three locked source slices")
    expected_counts = {"mali_real_qpu": 340, "qonductor_single_circuit_ibm": 4481,
                       "qpack_mcp": 3945}
    if any(int(by_source[source]["shared_success_n"]) != expected_counts[source]
           for source in order):
        raise ValueError("paired row denominators differ from the locked brief")

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(14.2, 6.2), gridspec_kw={"width_ratios": [1.22, 1]},
    )
    title_block(fig, "Real-QPU source-local paired error",
                "V3-large graph adaptation vs unified polynomial · lower MAE is better · same successful test IDs per source")
    yy = list(range(len(order)))[::-1]
    for y, source in zip(yy, order):
        row = by_source[source]
        graph = float(row["candidate_mae_shared"])
        poly = float(row["reference_mae_shared"])
        delta = float(row["paired_mae_delta_candidate_minus_reference"])
        lo = float(row["bootstrap_pooled_delta_ci_low"])
        hi = float(row["bootstrap_pooled_delta_ci_high"])
        n = int(row["shared_success_n"])
        left.plot([graph, poly], [y, y], color="#B8C5D3", linewidth=2, zorder=1)
        left.scatter(graph, y, s=68, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
        left.scatter(poly, y, s=68, color=ORANGE, edgecolor="white", linewidth=0.8, zorder=3)
        left.text(max(graph, poly) + 0.025, y, f"n={n:,}", va="center", fontsize=8, color=MUTED)
        right.errorbar(delta, y, xerr=[[delta - lo], [hi - delta]], fmt="o", color=BLUE,
                       ecolor=BLUE, elinewidth=2, capsize=4, markersize=6, zorder=3)
        right.text(hi + 0.025, y, f"{delta:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                   va="center", fontsize=8, color=INK)
    left.set_yticks(yy, [labels[source] for source in order])
    left.set_xlabel("MAE on the shared test rows (seconds)", color=MUTED)
    left.set_xlim(0, 1.75)
    left.set_title("Per-method MAE", loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.legend(handles=[Patch(color=BLUE, label="V3-large graph adaptation"),
                        Patch(color=ORANGE, label="Polynomial adaptation")],
               frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.115),
               ncol=2, fontsize=8)
    right.axvline(0, color=INK, linewidth=1, linestyle="--", alpha=0.8)
    right.set_yticks(yy, [labels[source] for source in order])
    right.set_xlabel("Paired ΔMAE = graph − polynomial (seconds)", color=MUTED)
    right.set_xlim(-0.55, 0.13)
    right.set_title("Paired difference with 95% grouped-bootstrap CI", loc="left",
                    color=INK, fontsize=11, fontweight="bold")
    for ax in (left, right):
        configure_axes(ax)
        ax.tick_params(axis="y", length=0)
    fig.text(0.06, 0.082,
             "Source slices are descriptive, not separate source-trained models. Ma–Li CI crosses zero; Qonductor coverage is asymmetric (row4477 unavailable).",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=[0.04, 0.18, 0.98, 0.90])
    source = source_pin(path)
    description = (
        "Units: seconds. Row set: the three source rows for candidate=mali_graph_architecture_v3_large "
        "and reference=unified_polynomial_v3; shared n=340 Ma–Li, 4,481 Qonductor, 3,945 QPack. "
        "Each row's shared_row_set_hash is in the source CSV. Evaluation/output clock: "
        "archived_observed_service_execution_time. Limits: source slices are not source-specific training; "
        "Ma–Li interval crosses zero, Qonductor coverage is asymmetric, and this does not isolate graph structure."
    )
    return {"title": "Real-QPU source-local paired error", "caption": description,
            "row_set": [{"source_id": source_id, "n": expected_counts[source_id],
                         "shared_row_set_hash": by_source[source_id]["shared_row_set_hash"]}
                        for source_id in order],
            "units": "seconds", "clock": "archived_observed_service_execution_time",
            "sources": [source],
            "files": save_pair(fig, "fig1_qpu_paired_mae_by_source", description, [source])}


def plot_qpu_coverage() -> dict:
    path = "qpu/archived_method_coverage.csv"
    rows = read_csv(path)
    if len(rows) != 24 or any(int(row["assigned_n"]) != 8767 for row in rows):
        raise ValueError("QPU coverage table is not 24 methods by 8,767 assigned rows")
    rows.sort(key=lambda r: (float(r["coverage"]), r["method_id"]))
    fig, ax = plt.subplots(figsize=(15.0, 12.0))
    title_block(fig, "Real-QPU prediction coverage",
                "24 evaluated method variants · 8,767 assigned archived observations per method · coverage is not accuracy")
    yy = list(range(len(rows)))
    for y, row in zip(yy, rows):
        coverage = float(row["coverage"])
        ax.barh(y, coverage, height=0.68, color=BLUE, zorder=2)
        ax.text(coverage + 0.012, y,
                f"{int(row['predicted_n']):,} / {int(row['assigned_n']):,}  ({coverage:.2%})",
                va="center", ha="left", fontsize=7.4, color=INK)
    ax.set_yticks(yy, [textwrap.fill(row["method_id"].replace("_", " "), 39) for row in rows])
    ax.set_xlim(0, 1.28)
    ax.set_xticks([0, .25, .5, .75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Successful prediction coverage (predicted / assigned)", color=MUTED)
    configure_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelsize=7.6)
    fig.text(0.06, 0.075,
             "Assigned denominator is retained, including unavailable/failure states. Variants target distinct outputs/clocks; bars must not be read as a quality or accuracy ranking.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=[0.28, 0.15, 0.83, 0.90])
    source = source_pin(path)
    description = (
        "Units: fraction of assigned observations, displayed as percent and counts. Exact row set: all "
        "24 rows in qpu/archived_method_coverage.csv; assigned denominator is 8,767 per method. "
        "Clock differs by method_output_clock as recorded in the table. Limit: coverage only, not accuracy; "
        "methods have different fidelity classes and output clocks, so no ranking is implied."
    )
    return {"title": "Real-QPU prediction coverage", "caption": description,
            "row_set": {"method_ids": [row["method_id"] for row in rows],
                        "method_rows": 24, "assigned_per_method": 8767},
            "units": "successful predictions / assigned observations; percent",
            "clock": "method-specific; see method_output_clock in the source table",
            "sources": [source],
            "files": save_pair(fig, "fig2_qpu_prediction_coverage", description, [source])}


def plot_simulator_mps_quality_gated() -> dict:
    metrics_path = "simulator/per_configuration_metrics.csv"
    availability_path = "simulator/availability_matrix.csv"
    metrics = [row for row in read_csv(metrics_path)
               if row["configuration_id"] == "cudaq_nvidia_mps_fp64"
               and row["engine_id"] == "cudaq_nvidia_mps"
               and row["quality_policy_id"] == "fp64_bond16_cutoff1e-10_gesvdj_fidelity_ge_0.99"
               and row["stratum"] in {"core_q2_q9", "frontier_q10_q16"}
               and row["evaluation_target_clock"] in {"first_execution", "warm_execution"}]
    if len(metrics) != 4:
        raise ValueError("expected exactly core/frontier × first/warm MPS metric rows")
    # The reader table includes paired target/output-clock rows for this native engine.
    # Keep only output==target observations; the same gate/config is then stratified by clock.
    selected = [row for row in metrics if row["method_output_clock"] == row["evaluation_target_clock"]]
    by_key = {(row["stratum"], row["evaluation_target_clock"]): row for row in selected}
    expected_keys = {(stratum, clock) for stratum in ("core_q2_q9", "frontier_q10_q16")
                     for clock in ("first_execution", "warm_execution")}
    if set(by_key) != expected_keys or len(selected) != 4:
        raise ValueError("MPS selected rows do not cover exactly core/frontier × first/warm")
    availability = [row for row in read_csv(availability_path)
                    if row["configuration_id"] == "cudaq_nvidia_mps_fp64"
                    and row["evaluation_target_clock"] in {"first_execution", "warm_execution"}]
    avail_by_clock = {row["evaluation_target_clock"]: row for row in availability}
    if set(avail_by_clock) != {"first_execution", "warm_execution"}:
        raise ValueError("MPS availability denominators missing clock rows")

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 6.0), sharey=True)
    title_block(fig, "CUDA-Q MPS under a fixed quality gate",
                "Same engine/configuration/precision; clocks and panel strata remain separate; failures stay in each denominator")
    clock_order = ["first_execution", "warm_execution"]
    labels = ["Core q2–q9", "Frontier q10–q16"]
    colors = {"core_q2_q9": BLUE, "frontier_q10_q16": TEAL}
    for ax, clock, heading in zip((left, right), clock_order, ["First execution", "Warm execution"]):
        configure_axes(ax)
        positions = [1, 0]
        for position, stratum, label in zip(positions, ("core_q2_q9", "frontier_q10_q16"), labels):
            row = by_key[(stratum, clock)]
            median_ms = float(row["median_seconds"]) * 1000.0
            attempted = int(row["attempted_cells"])
            ok = int(row["ok_cells"])
            fails = int(row["failure_cells"])
            ax.barh(position, median_ms, height=0.48, color=colors[stratum], zorder=2)
            ax.text(median_ms + 3.5, position,
                    f"{median_ms:.1f} ms  ·  {ok}/{attempted} ok, {fails} failures",
                    va="center", fontsize=8, color=INK)
        ax.set_title(heading, loc="left", color=INK, fontsize=11, fontweight="bold")
        ax.set_yticks(positions, labels)
        ax.set_xlabel("Median per session-cell (ms)", color=MUTED)
        ax.set_xlim(0, 330)
        ax.tick_params(axis="y", length=0)
    left.grid(axis="y", visible=False)
    right.grid(axis="y", visible=False)
    fig.text(0.06, 0.082,
             "Quality policy: fp64, bond 16, cutoff 1e−10, gesvdj, fidelity ≥0.99. Medians summarize successful rows; failed cells remain in the denominators.",
             color=MUTED, fontsize=8)
    fig.text(0.06, 0.057,
             "Descriptive within-engine/configuration comparison only; no cross-engine ranking or circuit-level paired claim.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=[0.05, 0.14, 0.98, 0.90])
    sources = [source_pin(metrics_path), source_pin(availability_path)]
    description = (
        "Units: milliseconds per successful session-cell median. Exact row set: CUDA-Q MPS fp64, "
        "quality policy fp64_bond16_cutoff1e-10_gesvdj_fidelity_ge_0.99; four per_configuration_metrics "
        "rows where method_output_clock equals evaluation_target_clock, crossing core/frontier with "
        "first/warm clock. Denominators are 486 core and 126 frontier session-cells per clock; availability "
        "table reports 462/486 and 111/126 successful cells per clock. Limit: medians are among successful "
        "rows; failure counts include all terminal failure classes and no engine ranking is made."
    )
    return {"title": "CUDA-Q MPS under a fixed quality gate", "caption": description,
            "row_set": [
                {"configuration_id": "cudaq_nvidia_mps_fp64", "stratum": key[0], "clock": key[1],
                 "attempted_cells": int(row["attempted_cells"]), "ok_cells": int(row["ok_cells"]),
                 "failure_cells": int(row["failure_cells"])}
                for key, row in sorted(by_key.items())],
            "units": "milliseconds per successful session-cell median",
            "clock": "first_execution and warm_execution shown separately",
            "quality_policy": "fp64_bond16_cutoff1e-10_gesvdj_fidelity_ge_0.99",
            "sources": sources,
            "files": save_pair(fig, "fig3_cudaq_mps_quality_gated_clocks", description, sources)}


def plot_maestro_pilot_spread() -> dict:
    path = "simulator/maestro_pilot_summary.csv"
    rows = read_csv(path)
    if len(rows) != 10 or any(int(row["timed_n"]) != 15 for row in rows):
        raise ValueError("Maestro pilot summary must contain 10 cells × 15 timed observations")
    # Keep frozen input order: the first nine are SV gate failures; final is the MPS gate pass.
    fig, ax = plt.subplots(figsize=(13.2, 8.0))
    title_block(fig, "Maestro pilot timing spread",
                "10 synthetic calibration cells · process-isolated reported execution · timing stability gate failed overall")
    colors = {"statevector": BLUE, "mps_fixed_chi": TEAL}
    labels = []
    x_min = math.inf
    x_max = 0.0
    for index, row in enumerate(rows):
        candidate = row["candidate"]
        repeats = int(row["operation_repeats"])
        op = {"one_qubit_noncommuting": "1Q", "two_qubit_wrapper_control": "2Q wrapper",
              "two_qubit_cx_interleaved": "2Q CX"}.get(row["operation_class"], row["operation_class"])
        status = "PASS" if row["stability_status"] == "ok" else "FAIL"
        prefix = "SV" if candidate == "statevector" else "MPS χ=32"
        labels.append(f"{prefix} · {op} × {repeats} · {status}")
        low = float(row["min_reported_seconds"])
        high = float(row["max_reported_seconds"])
        x_min, x_max = min(x_min, low), max(x_max, high)
        y = len(rows) - 1 - index
        color = colors[candidate]
        ax.hlines(y, low, high, color=color, linewidth=2.2, alpha=0.72, zorder=1)
        ax.scatter([low, high], [y, y], color=color, s=24, zorder=2)
        medians = json.loads(row["session_medians_seconds"])
        for session_index, median in enumerate(medians):
            ax.scatter(float(median), y, marker=["o", "s", "D"][session_index],
                       s=37, color=[BLUE, ORANGE, PURPLE][session_index],
                       edgecolor="white", linewidth=0.55, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(x_min * 0.72, x_max * 1.42)
    ax.set_yticks(range(len(rows)), labels[::-1])
    ax.set_xlabel("Reported timing (seconds; log scale)", color=MUTED)
    ax.set_title("Min–max observed range; markers are session medians", loc="left",
                 color=INK, fontsize=11, fontweight="bold")
    configure_axes(ax)
    ax.grid(axis="x", which="minor", color=GRID, linewidth=0.45)
    ax.tick_params(axis="y", length=0, labelsize=8.2)
    legend = [Line2D([0], [0], color="none", marker="o", markerfacecolor=BLUE,
                     markeredgecolor="white", label="Session 1 median"),
              Line2D([0], [0], color="none", marker="s", markerfacecolor=ORANGE,
                     markeredgecolor="white", label="Session 2 median"),
              Line2D([0], [0], color="none", marker="D", markerfacecolor=PURPLE,
                     markeredgecolor="white", label="Session 3 median"),
              Patch(color=TEAL, label="MPS cell passed stability gate"),
              Patch(color=BLUE, label="Statevector cells failed stability gate")]
    ax.legend(handles=legend, frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.52, -0.22), fontsize=8)
    fig.text(0.06, 0.07,
             "Terminal pilot: 9/9 statevector cells failed stability; 1/1 MPS cell passed. No accepted predictor or panel accuracy score; cause of spread is undetermined.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=[0.05, 0.15, 0.98, 0.90])
    source = source_pin(path)
    description = (
        "Units: seconds, logarithmic x-axis. Exact row set: all 10 rows in "
        "simulator/maestro_pilot_summary.csv, each with timed_n=15; row identity is cell_id. "
        "Clock: maestro_qcsim_process_isolated_reported_execution. Lines show min_reported_seconds to "
        "max_reported_seconds; symbols show the three session_medians_seconds. Limits: this was a "
        "stability pilot (9 statevector failures, 1 MPS pass), not predictor accuracy or common-panel execution; "
        "cause of timing spread was not established."
    )
    return {"title": "Maestro pilot timing spread", "caption": description,
            "row_set": [{"cell_id": row["cell_id"], "candidate": row["candidate"],
                         "operation_class": row["operation_class"],
                         "operation_repeats": int(row["operation_repeats"]),
                         "stability_status": row["stability_status"]}
                        for row in rows],
            "units": "seconds; logarithmic x-axis",
            "clock": "maestro_qcsim_process_isolated_reported_execution",
            "sources": [source],
            "files": save_pair(fig, "fig4_maestro_pilot_timing_spread", description, [source])}


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="work/presentation-figures")
    args = parser.parse_args()
    OUT = (ROOT / args.output_dir).resolve()
    OUT.relative_to(ROOT / "work")
    if OUT.exists():
        raise FileExistsError(f"choose a new output directory: {OUT}")
    if not PRESENTATION.is_file():
        raise FileNotFoundError(f"staged locked presentation claims not found: {PRESENTATION}")
    presentation = PRESENTATION.read_text(encoding="utf-8")
    for locked_phrase in ("CI Ma–Li qua 0", "không gộp những clock khác nhau",
                          "Status `pilot_gate_failed`"):
        if locked_phrase not in presentation:
            raise ValueError(f"staged presentation brief changed/missing locked phrase: {locked_phrase}")
    presentation_hash = sha256(PRESENTATION)
    metadata = [plot_qpu_paired_mae(), plot_qpu_coverage(),
                plot_simulator_mps_quality_gated(), plot_maestro_pilot_spread()]
    readme = [
        "# F-C4 figure source and captions",
        "",
        "These four figures were rendered only from the staged `two_domain_scorecard_v3` tables and the staged locked `docs/presentation.md`; no raw or prediction files were opened. They are internal review figures, not a cross-domain or cross-clock leaderboard.",
        "",
        f"Plot source: `{Path(__file__).resolve().relative_to(ROOT)}` SHA-256 `{sha256(Path(__file__).resolve())}`.",
        f"Locked-claims source: `docs/presentation.md` SHA-256 `{presentation_hash}`.",
        "",
    ]
    for index, item in enumerate(metadata, 1):
        readme.extend([
            f"## Figure {index}: {item['title']}", "",
            f"Files: `{item['files']['png']}` (SHA-256 `{item['files']['png_sha256']}`), `"
            f"{item['files']['svg']}` (SHA-256 `{item['files']['svg_sha256']}`).",
            f"Units: {item['units']}", f"Clock: {item['clock']}",
            f"Exact row set/denominator: `{json.dumps(item['row_set'], ensure_ascii=False, sort_keys=True)}`",
            f"Caption and limitations: {item['caption']}", "Sources:",
        ])
        for source in item["sources"]:
            readme.append(f"- `{source['path']}` — SHA-256 `{source['sha256']}`")
        if "quality_policy" in item:
            readme.append(f"Quality policy: `{item['quality_policy']}`")
        readme.append("")
    readme.extend([
        "## Reproduction",
        "",
        "From repository root, using the frozen report environment:",
        "",
        "```bash",
        "python presentations/source/plot_report_figures.py --output-dir work/presentation-figures",
        "```",
        "",
        "The script reads the committed scorecard tables and presentation brief and writes only to the new work output directory.",
        "",
    ])
    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps({"status": "figures_written", "count": len(metadata),
                      "presentation_sha256": presentation_hash,
                      "figures": [item["files"] for item in metadata]}, indent=2))


if __name__ == "__main__":
    main()
