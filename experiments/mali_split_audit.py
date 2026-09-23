#!/usr/bin/env python3
"""Create leakage-safe Ma--Li circuit/family split manifests.

The public Ma--Li labels contain two device CSVs and a directory of logical
QASM files.  This audit treats the raw QASM bytes as the identity of a
circuit.  Filename tokens are retained as metadata, but family holdout uses
connected components of the family--QASM graph so that aliases of identical
circuits cannot cross a split.

This script does not transpile, train, or contact IBM services.  It only writes
portable split metadata and a short audit report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        self.add(value)
        parent = self.parent[value]
        if parent != value:
            parent = self.find(parent)
            self.parent[value] = parent
        return parent

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def resolve_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    candidates = (
        ROOT / "work" / "mali",
        ROOT.parent / "Quantum-Execution-Time-Prediction",
    )
    for candidate in candidates:
        if (candidate / "data" / "quantum_circuits").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Ma--Li checkout not found; pass --mali-root")


def family_token(name: str) -> str:
    return str(name).lower().split("_indep_")[0].replace("-", "_")


def load_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for backend in ("osaka", "kyoto"):
        path = root / "data" / f"{backend}_time_taken.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for source in csv.DictReader(handle):
                circuit = source.get("quantum_circuit") or source.get("circuit_name")
                if not circuit:
                    raise ValueError(f"missing quantum_circuit in {path}")
                qasm_path = root / "data" / "quantum_circuits" / f"{circuit}.qasm"
                if not qasm_path.is_file():
                    raise FileNotFoundError(qasm_path)
                digest = hashlib.sha256(qasm_path.read_bytes()).hexdigest()
                rows.append(
                    {
                        "row_id": f"{backend}:{len(rows)}",
                        "backend": backend,
                        "circuit": circuit,
                        "qasm_path": str(qasm_path.relative_to(root)),
                        "qasm_sha256": digest,
                        "filename_family": family_token(circuit),
                        "target_seconds": float(source["time_taken"]),
                    }
                )
    return rows


def family_components(rows: list[dict[str, object]]) -> dict[str, str]:
    """Return hash -> stable family-component label."""
    uf = UnionFind()
    for row in rows:
        family = str(row["filename_family"])
        digest = str(row["qasm_sha256"])
        uf.union("family:" + family, "hash:" + digest)

    groups: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        family = str(row["filename_family"])
        digest = str(row["qasm_sha256"])
        root = uf.find("hash:" + digest)
        entry = groups.setdefault(root, {"families": set(), "hashes": set()})
        entry["families"].add(family)
        entry["hashes"].add(digest)

    labels: dict[str, str] = {}
    for entry in groups.values():
        families = sorted(entry["families"])
        label = "+".join(families)
        for digest in entry["hashes"]:
            labels[digest] = label
    return labels


def assign_group_folds(rows: list[dict[str, object]], seed: int, folds: int) -> dict[str, int]:
    groups = np.asarray(sorted({str(row["qasm_sha256"]) for row in rows}), dtype=object)
    rng = np.random.default_rng(seed)
    shuffled = groups[rng.permutation(len(groups))]
    assignment: dict[str, int] = {}
    for fold, bucket in enumerate(np.array_split(shuffled, folds), start=1):
        for digest in bucket:
            assignment[str(digest)] = fold
    return assignment


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def make_report(summary: dict[str, object]) -> str:
    lines = [
        "# Ma--Li leakage-safe split audit",
        "",
        f"Generated: **{summary['generated_utc']}**",
        "",
        "This manifest uses SHA-256 of raw QASM bytes as circuit identity. It is",
        "a data/split audit; it does not retrain a model or reconstruct a historical",
        "IBM calibration snapshot.",
        "",
        "## Inventory",
        "",
        f"- label rows: **{summary['n_rows']}** ({summary['backend_rows']})",
        f"- exact logical-QASM hashes: **{summary['n_qasm_hashes']}**",
        f"- hashes present on both backends: **{summary['paired_hashes']}**",
        f"- distinct `(QASM hash, backend)` cells: **{summary['qasm_backend_cells']}**",
        f"- duplicate cells under filename aliases: **{summary['duplicate_cells']}**",
        f"- family connected components: **{summary['n_family_components']}**",
        "",
        "## Primary grouped folds",
        "",
        "All backend copies and filename aliases of one QASM hash stay in the",
        "same outer fold. The fold number is deterministic for the recorded seed.",
        "",
        "| Fold | QASM groups | Rows | Osaka | Kyoto |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["grouped_fold_counts"]:
        lines.append(
            f"| {row['fold']} | {row['qasm_groups']} | {row['rows']} | "
            f"{row['osaka']} | {row['kyoto']} |"
        )
    lines.extend(
        [
            "",
            "## Strict backend + unseen-QASM diagnostic",
            "",
            "For each held backend and grouped fold, test rows are held-backend rows",
            "in that QASM fold. Training rows are the other backend with all QASM",
            "groups in the test fold removed. This is a two-backend diagnostic, not",
            "broad vendor generalization.",
            "",
            "| Held backend | Fold | Test rows | Train rows | Hash overlap |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["strict_splits"]:
        lines.append(
            f"| {row['held_backend']} | {row['fold']} | {row['test_rows']} | "
            f"{row['train_rows']} | {row['hash_overlap']} |"
        )
    lines.extend(
        [
            "",
            "## Family components",
            "",
            "Filename families are connected through exact QASM hashes before",
            "holdout. This prevents a byte-identical circuit named under two family",
            "tokens from crossing a family split.",
            "",
            "| Component | Families | Hashes | Rows |",
            "|---|---|---:|---:|",
        ]
    )
    for row in summary["family_components"]:
        lines.append(
            f"| `{row['component']}` | `{row['families']}` | {row['qasm_hashes']} | {row['rows']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The old row-level split can measure label adaptation when the same",
            "logical circuit appears on both devices, but it cannot support an",
            "unseen-circuit claim. Use `row_manifest.csv` to drive every later model",
            "evaluation. Simulator pretraining must also exclude outer-test QASM",
            "hashes when reporting strict all-domain transfer.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mali-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "validation" / "mali_deep_protocol_v1" / "split_audit",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    root = resolve_root(str(args.mali_root) if args.mali_root else None)
    rows = load_rows(root)
    components = family_components(rows)
    fold_by_hash = assign_group_folds(rows, args.seed, args.folds)
    for row in rows:
        digest = str(row["qasm_sha256"])
        row["family_component"] = components[digest]
        row["group_fold"] = fold_by_hash[digest]

    hash_to_backends: dict[str, set[str]] = {}
    cell_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        digest = str(row["qasm_sha256"])
        backend = str(row["backend"])
        hash_to_backends.setdefault(digest, set()).add(backend)
        cell = (digest, backend)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1

    grouped_counts = []
    for fold in range(1, args.folds + 1):
        subset = [row for row in rows if int(row["group_fold"]) == fold]
        grouped_counts.append(
            {
                "fold": fold,
                "qasm_groups": len({str(row["qasm_sha256"]) for row in subset}),
                "rows": len(subset),
                "osaka": sum(row["backend"] == "osaka" for row in subset),
                "kyoto": sum(row["backend"] == "kyoto" for row in subset),
            }
        )

    strict_splits = []
    for held_backend in ("osaka", "kyoto"):
        for fold in range(1, args.folds + 1):
            test = [
                row for row in rows
                if row["backend"] == held_backend and int(row["group_fold"]) == fold
            ]
            train = [
                row for row in rows
                if row["backend"] != held_backend and int(row["group_fold"]) != fold
            ]
            test_hashes = {str(row["qasm_sha256"]) for row in test}
            train_hashes = {str(row["qasm_sha256"]) for row in train}
            strict_splits.append(
                {
                    "held_backend": held_backend,
                    "fold": fold,
                    "test_rows": len(test),
                    "train_rows": len(train),
                    "test_hashes": len(test_hashes),
                    "train_hashes": len(train_hashes),
                    "hash_overlap": len(test_hashes & train_hashes),
                }
            )

    component_rows: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        component_rows.setdefault(str(row["family_component"]), []).append(row)
    component_summary = []
    for component in sorted(component_rows):
        subset = component_rows[component]
        component_summary.append(
            {
                "component": component,
                "families": "+".join(sorted({str(row["filename_family"]) for row in subset})),
                "qasm_hashes": len({str(row["qasm_sha256"]) for row in subset}),
                "rows": len(subset),
            }
        )

    summary = {
        "generated_utc": "2026-09-21",
        "protocol_date": "2026-09-21",
        "mali_root_relative": str(root),
        "seed": args.seed,
        "folds": args.folds,
        "n_rows": len(rows),
        "backend_rows": ", ".join(
            f"{backend}={sum(row['backend'] == backend for row in rows)}"
            for backend in ("osaka", "kyoto")
        ),
        "n_qasm_hashes": len(hash_to_backends),
        "paired_hashes": sum(len(backends) == 2 for backends in hash_to_backends.values()),
        "qasm_backend_cells": len(cell_counts),
        "duplicate_cells": sum(count > 1 for count in cell_counts.values()),
        "n_family_components": len(component_summary),
        "grouped_fold_counts": grouped_counts,
        "strict_splits": strict_splits,
        "family_components": component_summary,
        "strict_hash_overlap_max": max(row["hash_overlap"] for row in strict_splits),
        "target_semantics": (
            "Ma-Li upstream Osaka/Kyoto observed result.time_taken; 1024 shots; "
            "queue excluded according to upstream data contract"
        ),
        "limitations": [
            "raw CSVs do not contain historical job IDs or calibration timestamps",
            "exact logical QASM identity is byte-level SHA-256",
            "family components are derived from filename family plus exact QASM overlap",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "row_manifest.csv", rows)
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(make_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
