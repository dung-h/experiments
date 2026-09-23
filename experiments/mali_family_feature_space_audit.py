#!/usr/bin/env python3
"""Experiment A: feature-space audit of Ma--Li selected vs missing MQT families.

This compares circuit structure, not observed QPU runtime. The 12 families
absent from the Osaka/Kyoto label table have no `result.time_taken` here, so
they cannot be scored as a runtime estimator. Physical features are
reconstructed from the current FakeOsaka/FakeKyoto snapshot and must be read
as CURRENT_FAKE_SNAPSHOT_PROXY, not historical job-day calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from mali_family_ood_common import (
    LOGICAL_FEATURE_KEYS,
    MISSING_FAMILIES,
    PHYSICAL_FEATURE_KEYS,
    SELECTED_FAMILIES,
    filename_family,
    group_of,
    ks_median,
    logical_features,
    nearest_centroid_group_accuracy,
    sha256_file,
    standardized_centroid_distance,
    two_qubit_depth,
    width_bucket,
    width_from_name,
)

LOGICAL_FIELDS = [
    "circuit",
    "qasm_path",
    "qasm_sha256",
    "family",
    "family_group",
    "logical_width",
    "width_bucket",
    "hw_sampled",
    *LOGICAL_FEATURE_KEYS,
    "status",
    "error",
]
PHYSICAL_FIELDS = [
    "circuit",
    "backend",
    "qasm_sha256",
    "family",
    "family_group",
    "logical_width",
    "width_bucket",
    "hw_sampled",
    *LOGICAL_FEATURE_KEYS,
    *PHYSICAL_FEATURE_KEYS,
    "unsupported_duration_ops",
    "transpile_seed",
    "optimization_level",
    "status",
    "error",
    "wall_seconds",
]


def resolve_mali_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    candidates = (
        ROOT / "work" / "mali",
        ROOT.parent / "Quantum-Execution-Time-Prediction",
    )
    for candidate in candidates:
        if (candidate / "data" / "quantum_circuits").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Ma-Li checkout with data/quantum_circuits not found")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_hw_hashes(split_manifest: Path) -> set[str]:
    return {row["qasm_sha256"] for row in read_csv(split_manifest)}


def index_qasms(mali_root: Path) -> list[dict[str, object]]:
    directory = mali_root / "data" / "quantum_circuits"
    rows = []
    for path in sorted(directory.glob("*.qasm")):
        family = filename_family(path.name)
        width = width_from_name(path.name)
        rows.append(
            {
                "circuit": path.stem,
                "qasm_path": str(path.relative_to(mali_root)),
                "abs_path": path,
                "family": family,
                "family_group": group_of(family),
                "logical_width": width,
                "width_bucket": width_bucket(width),
            }
        )
    return rows


def compute_logical_row(item: dict[str, object], hw_hashes: set[str]) -> dict[str, object]:
    from qiskit import QuantumCircuit

    path = Path(item["abs_path"])
    digest = sha256_file(path)
    row = {
        "circuit": item["circuit"],
        "qasm_path": item["qasm_path"],
        "qasm_sha256": digest,
        "family": item["family"],
        "family_group": item["family_group"],
        "logical_width": item["logical_width"],
        "width_bucket": item["width_bucket"],
        "hw_sampled": digest in hw_hashes,
        "status": "ok",
        "error": "",
    }
    try:
        circuit = QuantumCircuit.from_qasm_file(str(path))
        row.update(logical_features(circuit))
        row["logical_width"] = int(circuit.num_qubits)
        row["width_bucket"] = width_bucket(int(circuit.num_qubits))
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        for key in LOGICAL_FEATURE_KEYS:
            row.setdefault(key, "")
    return row


_WORKER_BACKENDS = {}


def _init_physical_worker() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd() / "experiments"))
    from qiskit_ibm_runtime.fake_provider import FakeKyoto, FakeOsaka

    global _WORKER_BACKENDS
    _WORKER_BACKENDS = {"osaka": FakeOsaka(), "kyoto": FakeKyoto()}


def _physical_worker(payload: dict[str, object]) -> dict[str, object]:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd() / "experiments"))
    from qiskit import QuantumCircuit, transpile
    from mali_qcre_proxy import weighted_critical_path

    started = time.time()
    backend_name = str(payload["backend"])
    seed = int(payload["seed"])
    optimization_level = int(payload["optimization_level"])
    row = {
        "circuit": payload["circuit"],
        "backend": backend_name,
        "qasm_sha256": payload["qasm_sha256"],
        "family": payload["family"],
        "family_group": payload["family_group"],
        "logical_width": payload["logical_width"],
        "width_bucket": payload["width_bucket"],
        "hw_sampled": payload["hw_sampled"],
        "transpile_seed": seed,
        "optimization_level": optimization_level,
        "status": "ok",
        "error": "",
        "unsupported_duration_ops": "",
    }
    for key in LOGICAL_FEATURE_KEYS:
        row[key] = payload.get(key, "")
    try:
        original = QuantumCircuit.from_qasm_file(str(payload["abs_path"]))
        backend = _WORKER_BACKENDS[backend_name]
        compiled = transpile(
            original,
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=seed,
        )
        weighted_dt, unsupported = weighted_critical_path(compiled, backend)
        counts = compiled.count_ops()
        physical_two = sum(
            count for name, count in counts.items() if name in {"cx", "cz", "ecr", "swap"}
        )
        logical_depth = float(row["logical_depth"] or original.depth() or 0)
        logical_two = float(row["logical_two_qubit_count"] or 0.0)
        physical_depth = float(compiled.depth() or 0)
        row.update(
            {
                "physical_depth": physical_depth,
                "physical_two_qubit_depth": float(two_qubit_depth(compiled)),
                "physical_two_qubit_gate_count": float(physical_two),
                "physical_swap_count": float(counts.get("swap", 0)),
                "physical_gate_count": float(len(compiled.data)),
                "qcre_weighted_critical_path_seconds": float(weighted_dt) * float(backend.dt),
                "routing_depth_ratio": (physical_depth / logical_depth) if logical_depth else 0.0,
                "routing_two_qubit_ratio": (float(physical_two) / logical_two) if logical_two else 0.0,
                "unsupported_duration_ops": ",".join(sorted(set(unsupported))),
            }
        )
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["error_trace"] = traceback.format_exc(limit=3)
        for key in PHYSICAL_FEATURE_KEYS:
            row.setdefault(key, "")
    row["wall_seconds"] = round(time.time() - started, 4)
    return row


def feature_matrix(rows: list[dict[str, object]], keys: tuple[str, ...]) -> tuple[np.ndarray, list[str], list[str]]:
    usable = []
    families = []
    groups = []
    for row in rows:
        if str(row.get("status", "ok")) != "ok":
            continue
        try:
            usable.append([float(row[key]) for key in keys])
        except (TypeError, ValueError):
            continue
        families.append(str(row["family"]))
        groups.append(str(row["family_group"]))
    return np.asarray(usable, dtype=float), families, groups


def split_by(rows: list[dict[str, object]], key: str, left, right) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    a = [row for row in rows if row.get(key) in left]
    b = [row for row in rows if row.get(key) in right]
    return a, b


def compare_blocks(
    left: list[dict[str, object]],
    right: list[dict[str, object]],
    keys: tuple[str, ...],
    left_name: str,
    right_name: str,
) -> list[dict[str, object]]:
    output = []
    for feature in keys:
        try:
            left_values = np.asarray([float(row[feature]) for row in left if str(row.get("status", "ok")) == "ok"], dtype=float)
            right_values = np.asarray([float(row[feature]) for row in right if str(row.get("status", "ok")) == "ok"], dtype=float)
        except (TypeError, ValueError):
            continue
        stats = ks_median(left_values, right_values)
        output.append(
            {
                "comparison": f"{left_name}_vs_{right_name}",
                "feature": feature,
                **stats,
            }
        )
    return output


def family_medians(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    families = sorted({str(row["family"]) for row in rows})
    output = []
    for family in families:
        subset = [row for row in rows if row["family"] == family and str(row.get("status", "ok")) == "ok"]
        item = {
            "family": family,
            "family_group": group_of(family),
            "n": len(subset),
        }
        for key in keys:
            values = []
            for row in subset:
                try:
                    values.append(float(row[key]))
                except (TypeError, ValueError):
                    continue
            item[f"median_{key}"] = float(np.median(values)) if values else float("nan")
        output.append(item)
    return output


def run_logical(items: list[dict[str, object]], hw_hashes: set[str], workers: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if workers <= 1:
        for index, item in enumerate(items, start=1):
            row = compute_logical_row(item, hw_hashes)
            rows.append(row)
            if index % 50 == 0 or index == len(items):
                print(f"logical {index}/{len(items)}", flush=True)
        return rows
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = [pool.submit(compute_logical_row, item, hw_hashes) for item in items]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 50 == 0 or index == len(items):
                print(f"logical {index}/{len(items)}", flush=True)
    rows.sort(key=lambda row: (str(row["family"]), int(row.get("logical_width") or 0), str(row["circuit"])))
    return rows


def run_physical(
    logical_rows: list[dict[str, object]],
    mali_root: Path,
    output_csv: Path,
    workers: int,
    seed: int,
    optimization_level: int,
    backends: tuple[str, ...],
) -> list[dict[str, object]]:
    existing = {
        (row["circuit"], row["backend"]): row
        for row in read_csv(output_csv)
        if row.get("status") in {"ok", "error", "timeout"}
    }
    jobs = []
    for row in logical_rows:
        if str(row.get("status")) != "ok":
            continue
        for backend in backends:
            key = (row["circuit"], backend)
            if key in existing:
                continue
            jobs.append(
                {
                    "circuit": row["circuit"],
                    "backend": backend,
                    "qasm_sha256": row["qasm_sha256"],
                    "family": row["family"],
                    "family_group": row["family_group"],
                    "logical_width": row["logical_width"],
                    "width_bucket": row["width_bucket"],
                    "hw_sampled": row["hw_sampled"],
                    "abs_path": str(mali_root / row["qasm_path"]),
                    "seed": seed,
                    "optimization_level": optimization_level,
                    **{key: row[key] for key in LOGICAL_FEATURE_KEYS},
                }
            )
    rows = list(existing.values())
    print(f"physical resume {len(existing)} existing, {len(jobs)} remaining", flush=True)
    if not jobs:
        return rows
    family_cost = {
        "random": 5,
        "qft": 4,
        "qftentangled": 4,
        "qpeexact": 4,
        "qpeinexact": 4,
        "qnn": 3,
        "su2random": 3,
        "realamprandom": 3,
        "twolocalrandom": 3,
        "grover-noancilla": 5,
        "grover-v-chain": 5,
    }
    jobs.sort(key=lambda job: (family_cost.get(str(job["family"]), 1), int(job["logical_width"])))
    ctx = get_context("spawn")
    completed = 0
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx, initializer=_init_physical_worker) as pool:
        futures = {pool.submit(_physical_worker, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                row = {
                    **{key: job.get(key, "") for key in PHYSICAL_FIELDS},
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "wall_seconds": "",
                }
            rows.append(row)
            completed += 1
            if completed % 10 == 0 or completed == len(jobs):
                write_csv(output_csv, rows, PHYSICAL_FIELDS)
                print(
                    f"physical {len(existing)+completed}/{len(existing)+len(jobs)} "
                    f"last={job['circuit']}:{job['backend']} status={row.get('status')}",
                    flush=True,
                )
    write_csv(output_csv, rows, PHYSICAL_FIELDS)
    return rows


def summarize_comparison(rows: list[dict[str, object]], keys: tuple[str, ...], label: str) -> dict[str, object]:
    selected = [row for row in rows if row.get("family_group") == "selected" and str(row.get("status", "ok")) == "ok"]
    missing = [row for row in rows if row.get("family_group") == "missing" and str(row.get("status", "ok")) == "ok"]
    sampled = [row for row in selected if str(row.get("hw_sampled")) in {"True", "true", "1"}]
    unsampled = [row for row in selected if str(row.get("hw_sampled")) not in {"True", "true", "1"}]
    feature_rows = compare_blocks(selected, missing, keys, "selected", "missing")
    feature_rows.extend(compare_blocks(sampled, unsampled, keys, "hw_sampled", "selected_unsampled"))
    bucket_rows = []
    for bucket, _lo, _hi in (("2-16", 2, 16), ("17-50", 17, 50), ("51-90", 51, 90), ("91-127", 91, 127)):
        left = [row for row in selected if row.get("width_bucket") == bucket]
        right = [row for row in missing if row.get("width_bucket") == bucket]
        if len(left) >= 5 and len(right) >= 5:
            bucket_rows.extend(compare_blocks(left, right, keys, f"selected_{bucket}", f"missing_{bucket}"))
    matrix, families, _groups = feature_matrix(rows, keys)
    centroid = {"centroid_euclidean": float("nan")}
    loo = {"accuracy": float("nan"), "n_families": 0, "rows": []}
    if len(matrix):
        selected_x = np.asarray([row for row, family in zip(matrix, families) if group_of(family) == "selected"])
        missing_x = np.asarray([row for row, family in zip(matrix, families) if group_of(family) == "missing"])
        log_matrix = np.log1p(np.maximum(matrix, 0.0))
        centroid = standardized_centroid_distance(
            np.log1p(np.maximum(selected_x, 0.0)),
            np.log1p(np.maximum(missing_x, 0.0)),
        )
        loo = nearest_centroid_group_accuracy(log_matrix, families)
    return {
        "label": label,
        "n_selected": len(selected),
        "n_missing": len(missing),
        "n_hw_sampled": len(sampled),
        "n_selected_unsampled": len(unsampled),
        "centroid": centroid,
        "leave_one_family_group_accuracy": loo["accuracy"],
        "leave_one_family_rows": loo["rows"],
        "feature_comparisons": feature_rows,
        "bucket_comparisons": bucket_rows,
        "family_medians": family_medians(rows, keys),
    }


def fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if np.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def render_report(summary: dict[str, object]) -> str:
    lines = [
        "# Ma--Li family feature-space audit",
        "",
        f"**Finding recorded:** {summary['finding_date']}.",
        "",
        "This is Experiment A: a structural comparison of the 10 MQT families",
        "that appear in the Ma--Li Osaka/Kyoto observed-runtime table against",
        "the 12 families present in the local MQT Bench extract but absent from",
        "that hardware table. It does **not** score a runtime estimator and it",
        "does not create labels for the missing families.",
        "",
        "Target semantics remain unchanged. The 340 hardware rows are observed",
        "`result.time_taken` on retired `ibm_osaka` / `ibm_kyoto` jobs. The 12",
        "missing families have no such labels here. Physical features, when",
        "present, are reconstructed from the current FakeOsaka/FakeKyoto",
        "snapshot (`CURRENT_FAKE_SNAPSHOT_PROXY`).",
        "",
        "## Coverage",
        "",
        f"- MQT QASM files indexed: {summary['n_qasm']}",
        f"- Selected families: {', '.join(SELECTED_FAMILIES)}",
        f"- Missing families: {', '.join(MISSING_FAMILIES)}",
        f"- Hardware-sampled QASM hashes: {summary['n_hw_hashes']}",
        f"- Logical rows ok: {summary['logical']['n_ok']}",
        f"- Physical rows ok: {summary['physical']['n_ok']}",
        f"- Physical errors/timeouts: {summary['physical']['n_bad']}",
        "",
        "Experiment B remains blocked: `ibm_osaka` retired 2024-08-13 and",
        "`ibm_kyoto` retired 2024-09-05, and this workspace has no",
        "`result.time_taken` for the 12 missing families.",
        "",
        "## Logical structure, selected vs missing",
        "",
        f"Standardized log1p centroid distance: **{fmt(summary['logical_audit']['centroid']['centroid_euclidean'])}**.",
        f"Leave-one-family nearest-group accuracy: **{fmt(summary['logical_audit']['leave_one_family_group_accuracy'], 3)}**",
        f"({summary['logical_audit']['n_selected']} selected circuits, {summary['logical_audit']['n_missing']} missing).",
        "",
        "| Feature | Median selected | Median missing | Ratio missing/selected | KS | p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["logical_audit"]["feature_comparisons"]:
        if row["comparison"] != "selected_vs_missing":
            continue
        lines.append(
            f"| `{row['feature']}` | {fmt(row['median_left'])} | {fmt(row['median_right'])} | "
            f"{fmt(row['median_ratio_right_over_left'])} | {fmt(row['ks_statistic'], 3)} | {fmt(row['ks_pvalue'], 4)} |"
        )
    lines.extend(
        [
            "",
            "### Width buckets",
            "",
            "The selected/missing gap is not only a different width mix. At 91--127",
            "qubits the two-qubit-count and interaction-density distributions do",
            "not overlap. At 2--16 qubits the same comparison is milder.",
            "",
            "| Bucket | Feature | Median selected | Median missing | KS |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["logical_audit"]["bucket_comparisons"]:
        if row["feature"] not in {"logical_depth", "logical_two_qubit_count", "interaction_density"}:
            continue
        bucket = row["comparison"].replace("selected_", "").replace("_vs_missing_", " vs ")
        # comparison is selected_2-16_vs_missing_2-16
        left, _, right = row["comparison"].partition("_vs_")
        bucket = left.replace("selected_", "")
        lines.append(
            f"| {bucket} | `{row['feature']}` | {fmt(row['median_left'])} | {fmt(row['median_right'])} | {fmt(row['ks_statistic'], 3)} |"
        )
    lines.extend(
        [
            "",
            "### Hardware sampling inside the 10 selected families",
            "",
            "The hardware table is not a uniform sample of those families. The",
            "comparison below is selected-and-sampled versus selected-but-unsampled.",
            "",
            "| Feature | Median HW-sampled | Median unsampled | Ratio unsampled/sampled | KS | p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["logical_audit"]["feature_comparisons"]:
        if row["comparison"] != "hw_sampled_vs_selected_unsampled":
            continue
        lines.append(
            f"| `{row['feature']}` | {fmt(row['median_left'])} | {fmt(row['median_right'])} | "
            f"{fmt(row['median_ratio_right_over_left'])} | {fmt(row['ks_statistic'], 3)} | {fmt(row['ks_pvalue'], 4)} |"
        )
    lines.extend(["", "### Leave-one-family group assignment", "", "| Family | True group | Predicted | d(selected) | d(missing) |", "|---|---|---|---:|---:|"])
    for row in summary["logical_audit"]["leave_one_family_rows"]:
        lines.append(
            f"| `{row['family']}` | {row['true_group']} | {row['predicted_group']} | "
            f"{fmt(row['distance_to_selected'])} | {fmt(row['distance_to_missing'])} |"
        )
    if summary.get("physical_audit"):
        audit = summary["physical_audit"]
        lines.extend(
            [
                "",
                "## Physical proxy structure, selected vs missing",
                "",
                "These columns use current FakeOsaka/FakeKyoto transpilation. They",
                "answer whether compiled depth, routing and duration-weighted path",
                "also separate the two family groups. They are not observed QPU",
                "runtimes.",
                "",
                f"Physical centroid distance: **{fmt(audit['centroid']['centroid_euclidean'])}**.",
                f"Leave-one-family nearest-group accuracy: **{fmt(audit['leave_one_family_group_accuracy'], 3)}**.",
                "",
                "| Feature | Median selected | Median missing | Ratio missing/selected | KS | p |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in audit["feature_comparisons"]:
            if row["comparison"] != "selected_vs_missing":
                continue
            lines.append(
                f"| `{row['feature']}` | {fmt(row['median_left'])} | {fmt(row['median_right'])} | "
                f"{fmt(row['median_ratio_right_over_left'])} | {fmt(row['ks_statistic'], 3)} | {fmt(row['ks_pvalue'], 4)} |"
            )
    lines.extend(
        [
            "",
            "## What this can and cannot claim",
            "",
            "- It can show whether the 12 missing families occupy a different",
            "  region of logical, and if computed, compiled feature space.",
            "- It can show that Ma--Li's 170 hardware QASMs are a biased subset",
            "  even of the 10 selected families.",
            "- It cannot convert that structural gap into a runtime score for",
            "  the 12 missing families.",
            "- Family-held-out metrics on the 340 rows remain a statement about",
            "  the 10 selected families only.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mali-root", type=str, default=None)
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/validation/mali_family_ood_v1/feature_space_audit")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--logical-only", action="store_true")
    parser.add_argument("--finding-date", default="2026-09-22")
    args = parser.parse_args()

    mali_root = resolve_mali_root(args.mali_root)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    hw_hashes = load_hw_hashes(args.split_manifest)
    items = index_qasms(mali_root)
    print(f"indexed {len(items)} qasm from {mali_root}", flush=True)

    logical_csv = output_dir / "logical_features.csv"
    cached = {
        row["circuit"]: row
        for row in read_csv(logical_csv)
        if row.get("status") == "ok"
    }
    pending = [item for item in items if item["circuit"] not in cached]
    if pending:
        computed = run_logical(pending, hw_hashes, args.workers)
        logical_rows = list(cached.values()) + computed
        write_csv(logical_csv, logical_rows, LOGICAL_FIELDS)
    else:
        logical_rows = list(cached.values())
        print(f"logical cache hit {len(logical_rows)}", flush=True)

    for row in logical_rows:
        row["hw_sampled"] = bool(row.get("hw_sampled") in {True, "True", "true", "1"} or row.get("qasm_sha256") in hw_hashes)
        row["family_group"] = group_of(str(row["family"]))

    physical_rows: list[dict[str, object]] = []
    physical_csv = output_dir / "physical_proxy_features.csv"
    if not args.logical_only:
        physical_rows = run_physical(
            logical_rows,
            mali_root,
            physical_csv,
            args.workers,
            args.seed,
            args.optimization_level,
            ("osaka", "kyoto"),
        )

    logical_ok = [row for row in logical_rows if str(row.get("status")) == "ok"]
    physical_ok = [row for row in physical_rows if str(row.get("status")) == "ok"]
    logical_audit = summarize_comparison(logical_ok, LOGICAL_FEATURE_KEYS, "logical")
    physical_audit = (
        summarize_comparison(physical_ok, (*LOGICAL_FEATURE_KEYS, *PHYSICAL_FEATURE_KEYS), "physical_proxy")
        if physical_ok else None
    )
    write_csv(output_dir / "logical_feature_comparisons.csv", logical_audit["feature_comparisons"] + logical_audit["bucket_comparisons"], sorted({*logical_audit["feature_comparisons"][0]} if logical_audit["feature_comparisons"] else {"comparison"}))
    if logical_audit["family_medians"]:
        write_csv(output_dir / "logical_family_medians.csv", logical_audit["family_medians"], list(logical_audit["family_medians"][0]))
    if physical_audit and physical_audit["feature_comparisons"]:
        write_csv(
            output_dir / "physical_feature_comparisons.csv",
            physical_audit["feature_comparisons"] + physical_audit["bucket_comparisons"],
            list(physical_audit["feature_comparisons"][0]),
        )
        write_csv(output_dir / "physical_family_medians.csv", physical_audit["family_medians"], list(physical_audit["family_medians"][0]))

    summary = {
        "finding_date": args.finding_date,
        "n_qasm": len(items),
        "n_hw_hashes": len(hw_hashes),
        "selected_families": list(SELECTED_FAMILIES),
        "missing_families": list(MISSING_FAMILIES),
        "target_semantics": "no runtime target in this experiment; structure only",
        "proxy_semantics": "physical features are CURRENT_FAKE_SNAPSHOT_PROXY FakeOsaka/FakeKyoto transpile + target-duration weighted path",
        "logical": {
            "n_ok": len(logical_ok),
            "n_bad": len(logical_rows) - len(logical_ok),
            "path": str(logical_csv.relative_to(ROOT)),
        },
        "physical": {
            "n_ok": len(physical_ok),
            "n_bad": len(physical_rows) - len(physical_ok),
            "path": str(physical_csv.relative_to(ROOT)) if physical_rows else None,
            "logical_only": bool(args.logical_only),
        },
        "logical_audit": logical_audit,
        "physical_audit": physical_audit,
        "experiment_b_status": "blocked_no_qpu_labels_for_missing_families",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    (output_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_qasm", "n_hw_hashes", "logical", "physical", "experiment_b_status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
