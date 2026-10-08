#!/usr/bin/env python3
"""Render two QPU figures from public aggregates; never fit or score raw data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results/figures"
SOURCES = ("results/real_qpu/dataset_profile.json",
           "results/real_qpu/shared_row_comparisons.csv")
IDS = ("dataset_composition", "paired_method_differences")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, folder, name):
    fig.savefig(folder / (name + ".png"), dpi=160,
                metadata={"Software": "Quantum Runtime Estimation Benchmark"})
    fig.savefig(folder / (name + ".svg"),
                metadata={"Date": None, "Creator": "Quantum Runtime Estimation Benchmark"})
    plt.close(fig)


def render(folder):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "quantum-runtime-benchmark-qpu"})
    profile = json.loads((ROOT / SOURCES[0]).read_text())
    names = ("mali_real_qpu", "qonductor_single_circuit_ibm", "qpack_mcp")
    labels = ("Ma–Li", "Qonductor", "QPack MCP")
    colors = ("#315A7D", "#BF7540", "#50836D")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, field, total, title in ((axes[0], "rows", 4515, "4,515 archived observations"),
                                  (axes[1], "groups", 165, "165 conservative groups")):
        values = [profile["source_profiles"][name][field] for name in names]
        bars = ax.barh(labels, values, color=colors, height=0.58)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.3)
        for bar, value in zip(bars, values):
            ax.text(value + max(values) * 0.035, bar.get_y() + bar.get_height() / 2,
                    f"{value:,} ({value / total:.1%})", va="center", fontsize=10)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Observations" if field == "rows" else "Groups")
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("Real-QPU dataset: repeated contexts are not independent circuits", x=0.04,
                 ha="left", fontsize=13, fontweight="bold")
    fig.text(0.04, 0.035, "Inputs: exact source logical / archive-supported recipe / representative-angle QAOA structure.",
             fontsize=9, color="#444444")
    fig.subplots_adjust(left=0.12, right=0.98, wspace=0.55, bottom=0.17, top=0.78)
    save(fig, folder, IDS[0])

    with (ROOT / SOURCES[1]).open(newline="") as stream:
        pairs = list(csv.DictReader(stream))
    comparisons = (("mali_global_mlp", "mali_logical_graph", "MLP − graph"),
                   ("mali_logical_graph", "qonductor_budgeted_selector", "Graph − selected regression"))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for y, (left, right, label) in enumerate(comparisons):
        found = [r for r in pairs if {r["left_method"], r["right_method"]} == {left, right}]
        if len(found) != 1:
            raise ValueError("paired comparison missing or duplicated")
        row = found[0]
        if int(row["n"]) != 4515 or int(row["groups"]) != 165:
            raise ValueError("paired figure population differs")
        point = float(row["observed_mae_delta_seconds"])
        lo, hi = float(row["bootstrap_ci_low_seconds"]), float(row["bootstrap_ci_high_seconds"])
        if row["left_method"] != left:
            point, lo, hi = -point, -hi, -lo
        ax.plot([lo, hi], [y, y], color=colors[y], linewidth=2.5)
        ax.plot([lo, hi], [y, y], "|", color=colors[y], markersize=13)
        ax.plot(point, y, "o", color=colors[y], markersize=8)
        ax.text(0, y + 0.28, f"{point:+.3f} s; 95% interval [{lo:+.3f}, {hi:+.3f}] s",
                ha="center", va="center", fontsize=10)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks([0, 1], [c[2] for c in comparisons])
    ax.set_ylim(1.65, -0.55)
    ax.set_xlim(-0.62, 0.62)
    ax.set_xlabel("Left method MAE − right method MAE (seconds)")
    ax.grid(axis="x", alpha=0.15)
    ax.set_title("Same 4,515 held-out observations: both comparisons remain unresolved", loc="left",
                 pad=24, fontsize=13, fontweight="bold")
    fig.text(0.04, 0.035, "165-group bootstrap; 10,000 replicates; seed 42. Intervals condition on saved fits, not refitting.",
             fontsize=9, color="#444444")
    fig.subplots_adjust(left=0.29, right=0.98, bottom=0.20, top=0.80)
    save(fig, folder, IDS[1])


def update_manifest():
    path = FIGURES / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifact_id"] = "benchmark-results-figures"
    manifest["scope"] = "Current 4515-observation QPU aggregates and unchanged simulator context figures"
    manifest["figures"] = [f for f in manifest["figures"] if f["figure_id"] not in IDS]
    for name in IDS:
        outputs = {suffix: {"path": f"results/figures/{name}.{suffix}",
                            "sha256": sha(FIGURES / f"{name}.{suffix}")}
                   for suffix in ("png", "svg")}
        manifest["figures"].append({"figure_id": name, "outputs": outputs,
            "source_paths": list(SOURCES), "counts": {"observations": 4515, "groups": 165},
            "interpretation": "Declared reconstruction-qualified cohort; no universal graph advantage",
            "units": "counts" if name == IDS[0] else "MAE difference in seconds"})
    manifest["figure_outputs"] = [out for figure in manifest["figures"] for out in figure["outputs"].values()]
    manifest["source_hashes_sha256"].update({name: sha(ROOT / name) for name in SOURCES})
    manifest["qpu_generator"] = {"path": "scripts/render_qpu_figures.py", "sha256": sha(Path(__file__)),
        "requirements": "requirements-figures.txt", "matplotlib": matplotlib.__version__}
    manifest["verification"] = "Inventory checks four saved figure pairs. QPU figures can be regenerated byte-for-byte with --check; simulator figures remain saved outputs."
    manifest["bootstrap_policy"] = "QPU plot uses saved observed deltas and 10000-replicate/seed42 grouped intervals; no new bootstrap or fitting."
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Render in a temporary directory and compare bytes")
    args = parser.parse_args()
    if args.check:
        with tempfile.TemporaryDirectory(prefix="quantum-runtime-figures-") as name:
            folder = Path(name)
            render(folder)
            for stem in IDS:
                for suffix in ("png", "svg"):
                    file = f"{stem}.{suffix}"
                    if sha(folder / file) != sha(FIGURES / file):
                        raise ValueError(f"figure bytes differ: {file}")
        print("Verified two QPU figure pairs from published aggregates; no fitting or measurement")
    else:
        render(FIGURES)
        update_manifest()
        print("Updated two QPU figure pairs and their manifest; simulator figure bytes unchanged")


if __name__ == "__main__":
    main()
