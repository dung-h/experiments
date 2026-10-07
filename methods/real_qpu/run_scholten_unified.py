#!/usr/bin/env python3
"""Recalculate unified nominal throughput from pinned compiled graphs.

No training, transpilation, circuit execution or historical-output overwrite.
The registry joins nominal inputs, not previously scored predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import pickle
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "benchmark_v1/protocol/scholten_unified_nominal.json"
OUTPUT = ROOT / "artifacts/real_qpu/scholten_nominal"
METRICS_CODE = ROOT / "benchmark_v1/scripts/build_analytical_review_outputs.py"
COMPARATORS = {
    "compiled_graph_mali_style_v1": "Compiled graph plus metadata adaptation",
    "compiled_metadata_mlp_v1": "Matched metadata MLP",
    "compiled_common_feature_polynomial_v1": "Compiled polynomial adaptation",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def indexed(rows, key):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {key}")
    return result


def normalize_backend(name):
    return name.replace("ibmq_", "ibm_", 1)


def wire_depth(graph):
    """Count unit-cost instructions; excluded directives still merge wires."""
    if graph.get("format") != "s71-operation-dag-v1":
        raise ValueError("unknown graph representation")
    n = int(graph["num_qubits"])
    depth = [0] * n
    names, arities = graph["names"], graph["arity"]
    if len(arities) != len(names):
        raise ValueError("operation/arity length mismatch")
    columns = [graph[f"q{j}"] for j in range(max(map(int, arities), default=0))]
    if any(len(column) != len(names) for column in columns):
        raise ValueError("wire-column length mismatch")
    for i, name in enumerate(names):
        if name in {"if_else", "for_loop", "while_loop", "switch_case"}:
            raise ValueError("dynamic control cannot be flattened")
        arity = int(arities[i])
        if arity < 0:
            raise ValueError("negative operation arity")
        wires = [int(columns[j][i]) for j in range(arity)]
        if any(q < 0 or q >= n for q in wires):
            raise ValueError("operation wire outside graph")
        end = max((depth[q] for q in wires), default=0)
        if wires and name not in {"barrier", "snapshot"}:
            end += 1
        for q in wires:
            depth[q] = end
    return max(depth, default=0)


def nominal_prediction(shots, depth, throughput):
    if int(shots) != float(shots) or int(shots) <= 0:
        raise ValueError("shots must be a positive integer")
    if int(depth) != depth or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    if not math.isfinite(float(throughput)) or float(throughput) <= 0:
        raise ValueError("throughput must be finite and positive")
    return int(shots) * int(depth) / float(throughput)


def inputs(protocol):
    for pin in protocol["input_pins"].values():
        if sha(ROOT / pin["path"]) != pin["sha256"]:
            raise ValueError(f"input hash mismatch: {pin['path']}")
    pins = protocol["input_pins"]
    canonical = indexed(read_csv(ROOT / pins["canonical"]["path"]), "canonical_row_id")
    outer = indexed(read_csv(ROOT / pins["outer"]["path"]), "canonical_observation_id")
    components = indexed(read_csv(ROOT / pins["components"]["path"]), "canonical_row_id")
    if set(canonical) != set(outer) or set(canonical) != set(components) or len(canonical) != 8767:
        raise ValueError("canonical/outer/component identity mismatch")
    for identity, c in components.items():
        raw = canonical[identity]
        if (c["source_id"] != raw["source_id"] or int(c["shots"]) != int(raw["shots"])
                or normalize_backend(c["backend"]) != normalize_backend(raw["backend"])
                or int(c["outer_fold"]) != int(outer[identity]["outer_fold"])
                or c["leakage_group"] != outer[identity]["unified_leakage_group_id"]):
            raise ValueError(f"component/canonical context mismatch: {identity}")
    hashes = json.loads((ROOT / pins["graph_hashes"]["path"]).read_text())
    return canonical, outer, components, hashes


def source_assets(protocol):
    """Verify snapshot fields before copying exact bytes as portable evidence."""
    assets = {}
    for backend, entry in protocol["entries"].items():
        nominal_prediction(1, 1, entry["value"])
        if entry["source_kind"] != "pinned_fake_backend_configuration":
            continue
        distribution = importlib.metadata.distribution(entry["package"])
        if distribution.version != entry["package_version"]:
            raise ValueError("throughput snapshot package version mismatch")
        path = Path(distribution.locate_file("qiskit_ibm_runtime/" + entry["package_resource"]))
        if sha(path) != entry["configuration_sha256"]:
            raise ValueError(f"throughput source asset mismatch: {backend}")
        data = json.loads(path.read_text())
        if data[entry["source_field"]] != entry["value"]:
            raise ValueError(f"throughput field mismatch: {backend}")
        assets[backend] = path.read_bytes()
    return assets


def load_metrics():
    spec = importlib.util.spec_from_file_location("scholten_metrics", METRICS_CODE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(output):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    protocol = json.loads(PROTOCOL.read_text())
    canonical, outer, components, hashes = inputs(protocol)
    assets = source_assets(protocol)
    metrics = load_metrics()
    cache = {}

    def depth_for(c):
        name = c["graph_file"]
        if not name or Path(name).name != name or hashes.get(name) != c["graph_sha256"]:
            raise ValueError("graph filename/hash registry mismatch")
        if name not in cache:
            data = (ROOT / protocol["graph_root"] / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != hashes[name]:
                raise ValueError(f"graph bytes mismatch: {name}")
            graph = pickle.loads(data)
            if len(graph["names"]) != int(c["operation_count"]):
                raise ValueError("saved operation count differs from graph")
            cache[name] = {"graph_file": name, "graph_sha256": hashes[name],
                           "operation_count": len(graph["names"]), "compiled_wire_depth": wire_depth(graph)}
        return cache[name]["compiled_wire_depth"]

    cells = defaultdict(list)
    for c in components.values():
        cells[(c["source_id"], normalize_backend(c["backend"]))].append(c)
    selected = set()
    for cell in cells.values():
        ordered = sorted(cell, key=lambda c: hashlib.sha256(c["canonical_row_id"].encode()).hexdigest())
        selected.update(c["canonical_row_id"] for c in ordered[:10])
        selected.add(max(cell, key=lambda c: int(c["operation_count"]))["canonical_row_id"])
    selected.add("qonductor_single_circuit_ibm|row4477")
    for identity in sorted(selected):
        c = components[identity]
        entry = protocol["entries"].get(normalize_backend(c["backend"]))
        if entry and c["graph_status"] == "available":
            nominal_prediction(c["shots"], depth_for(c), entry["value"])
    print(f"Preflight PASS: {len(selected)} observations, {len(cache)} graphs", flush=True)

    attempts = []
    for index, identity in enumerate(sorted(canonical), 1):
        raw, c = canonical[identity], components[identity]
        backend = normalize_backend(c["backend"])
        entry = protocol["entries"].get(backend)
        row = {"canonical_row_id": identity, "source_id": raw["source_id"], "backend": backend,
               "outer_fold": int(outer[identity]["outer_fold"]), "leakage_group": c["leakage_group"],
               "method_id": protocol["method_id"], "reader_label": protocol["reader_label"],
               "fidelity_class": protocol["fidelity_class"],
               "evaluation_target_clock": protocol["evaluation_target_clock"],
               "method_output_clock": protocol["method_output_clock"],
               "target_seconds": raw["target_seconds"], "shots": int(c["shots"]),
               "compiled_wire_depth": "", "nominal_CLOPS": entry["value"] if entry else "",
               "throughput_definition": entry["definition"] if entry else "",
               "throughput_source_kind": entry["source_kind"] if entry else "",
               "throughput_source_url": entry["source_url"] if entry else "",
               "historical_job_match": False, "graph_digest": c["graph_digest"],
               "prediction": "", "status": "unavailable", "terminal_reason": ""}
        if c["graph_status"] != "available":
            row["terminal_reason"] = "dynamic_control_no_frozen_branch_duration_semantics"
        elif entry is None:
            row["terminal_reason"] = "no_sourced_nominal_throughput"
        else:
            depth = depth_for(c)
            row.update(compiled_wire_depth=depth, prediction=nominal_prediction(c["shots"], depth, entry["value"]),
                       status="predicted")
        attempts.append(row)
        if index % 1000 == 0:
            print(f"Unified attempts {index}/8767; checked graphs {len(cache)}", flush=True)
    expected = protocol["expected"]
    valid = [r for r in attempts if r["status"] == "predicted"]
    source_count = Counter(r["source_id"] for r in valid)
    if len(valid) != expected["predicted"] or dict(source_count) != expected["source_predicted"]:
        raise ValueError(f"coverage differs from frozen expectation: {dict(source_count)}")
    source_results = []
    for source in sorted(expected["source_predicted"]):
        assigned = sum(r["source_id"] == source for r in attempts)
        source_results.append({"source_id": source, **metrics.summarize(
            [r for r in valid if r["source_id"] == source], assigned, "seconds")})
    summary = {"method_id": protocol["method_id"], "reader_label": protocol["reader_label"],
               **metrics.summarize(valid, 8767, "seconds"),
               "source_balanced_macro_mae_seconds": statistics.fmean(r["mae_seconds"] for r in source_results),
               "unavailable": 8767 - len(valid)}
    by_id = {r["canonical_row_id"]: r for r in valid}
    common = set(by_id)
    comparators = defaultdict(dict)
    for r in read_csv(ROOT / protocol["input_pins"]["reader_attempts"]["path"]):
        if r["method_id"] in COMPARATORS and r["status"] == "predicted":
            comparators[r["method_id"]][r["canonical_row_id"]] = r
    for method in COMPARATORS:
        common &= set(comparators[method])
    if not common:
        raise ValueError("empty common successful comparison")
    common_ids = sorted(common)
    common_hash = hashlib.sha256("\n".join(common_ids).encode()).hexdigest()
    comparisons = []
    for method, records in [(protocol["method_id"], by_id), *sorted(comparators.items())]:
        chosen = [records[i] for i in common_ids]
        for r in chosen:
            original = by_id[r["canonical_row_id"]]
            if (float(r["target_seconds"]) != float(original["target_seconds"])
                    or r["source_id"] != original["source_id"]
                    or int(r["outer_fold"]) != original["outer_fold"]):
                raise ValueError("same-row comparison target/source/fold mismatch")
        macro = statistics.fmean(metrics.summarize([r for r in chosen if r["source_id"] == s],
                                source_count[s], "seconds")["mae_seconds"] for s in source_count)
        comparisons.append({"method_id": method, "reader_label": COMPARATORS.get(method, protocol["reader_label"]),
                            "row_ids_sha256": common_hash, **metrics.summarize(chosen, 8767, "seconds"),
                            "source_balanced_macro_mae_seconds": macro,
                            "comparison_scope": "exploratory_same_rows_not_paper_exact_ranking"})

    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    for backend, data in assets.items():
        (output / "inputs" / f"{backend}_configuration.json").write_bytes(data)
    write_csv(output / "attempts.csv", attempts)
    write_csv(output / "method_metrics.csv", [summary])
    write_csv(output / "source_metrics.csv", source_results)
    write_csv(output / "same_row_comparison.csv", comparisons)
    write_csv(output / "graph_depths.csv", [cache[name] for name in sorted(cache)])
    write_csv(output / "throughput_registry.csv", [dict(backend=b, **e) for b, e in sorted(protocol["entries"].items())])
    report = ["# Unified nominal throughput benchmark", "", protocol["claim_boundary"], "",
              f"Predicted: {len(valid):,} / 8,767; unavailable: {8767-len(valid):,}.",
              f"Unavailable reasons: {dict(Counter(r['terminal_reason'] for r in attempts if r['status']!='predicted'))}.", "",
              f"MAE: {summary['mae_seconds']:.6f} s; R²: {summary['r2_seconds']:.6f}; "
              f"source-balanced MAE: {summary['source_balanced_macro_mae_seconds']:.6f} s.", "",
              "## Same successful observations", "",
              "These raw nominal proxy outputs and learned observed-time predictions have different output semantics.",
              "All metrics below use exactly the same successful observation IDs; no calibration was fitted.", "",
              "| Method | Observations | MAE (s) | R² | Source-balanced MAE (s) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for r in comparisons:
        report.append(f"| {r['reader_label']} | {r['n']} | {r['mae_seconds']:.6f} | "
                      f"{r['r2_seconds']:.6f} | {r['source_balanced_macro_mae_seconds']:.6f} |")
    report += ["", "Historical predictions remain unchanged. This result joins throughput inputs and recalculates",
               "every supported row from the same compiled-graph wire-depth rule, including measurement/reset/delay.",
               "Five snapshot entries explicitly expose CLOPS_v; Kolkata exposes unqualified clops;",
               "the three retained publication entries do not identify a CLOPS layer definition.",
               "These definitions are not converted or asserted equivalent. Brisbane uses the frozen reported 5,000",
               "value, not the conflicting later 180,000-CLOPS_h snapshot. Belem/Lima/Quito remain unverified here.", "",
               "Rebuild: `python benchmark_v1/scripts/run_scholten_unified.py --output-dir <new-directory>`.",
               "Verify: `python benchmark_v1/scripts/run_scholten_unified.py --validate --output-dir <directory>`."]
    (output / "README.md").write_text("\n".join(report) + "\n")
    manifest = {"method_id": protocol["method_id"], "status": "PASS", "assigned": 8767,
                "predicted": len(valid), "unavailable": 8767-len(valid),
                "preflight": {"selected_rows": len(selected), "row_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(selected)).encode()).hexdigest(), "status": "PASS"},
                "common_rows": len(common_ids), "common_row_ids_sha256": common_hash,
                "graph_files_checked": len(cache), "training_or_measurement_performed": False,
                "input_pins": protocol["input_pins"], "protocol_sha256": sha(PROTOCOL),
                "runner_sha256": sha(Path(__file__)), "metrics_code_sha256": sha(METRICS_CODE),
                "python": sys.version.split()[0], "output_hashes": {str(p.relative_to(output)): sha(p)
                    for p in sorted(output.rglob("*")) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return validate(output)


def validate(output):
    protocol = json.loads(PROTOCOL.read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    for path, expected in [(PROTOCOL, manifest["protocol_sha256"]),
                           (Path(__file__), manifest["runner_sha256"]),
                           (METRICS_CODE, manifest["metrics_code_sha256"])]:
        if sha(path) != expected:
            raise ValueError("code/protocol hash mismatch")
    canonical, outer, components, _ = inputs(protocol)
    for path, expected in manifest["output_hashes"].items():
        if sha(output / path) != expected:
            raise ValueError(f"output hash mismatch: {path}")
    rows = indexed(read_csv(output / "attempts.csv"), "canonical_row_id")
    if set(rows) != set(canonical):
        raise ValueError("unified attempt denominator mismatch")
    depths = indexed(read_csv(output / "graph_depths.csv"), "graph_file")
    reasons = Counter()
    for identity, r in rows.items():
        raw = canonical[identity]
        if (float(r["target_seconds"]) != float(raw["target_seconds"])
                or int(r["shots"]) != int(raw["shots"])
                or r["source_id"] != raw["source_id"]
                or int(r["outer_fold"]) != int(outer[identity]["outer_fold"])):
            raise ValueError("attempt target/shots/source/fold drift")
        if r["status"] != "predicted":
            if r["prediction"] != "":
                raise ValueError("unavailable prediction must be empty")
            reasons[r["terminal_reason"]] += 1
            continue
        entry = protocol["entries"][r["backend"]]
        d = depths[components[identity]["graph_file"]]
        independent = int(raw["shots"]) / float(entry["value"]) * int(d["compiled_wire_depth"])
        if not math.isclose(float(r["prediction"]), independent, rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError("independent nominal equation mismatch")
        if (int(r["compiled_wire_depth"]) != int(d["compiled_wire_depth"])
                or d["graph_sha256"] != components[identity]["graph_sha256"]
                or float(r["nominal_CLOPS"]) != float(entry["value"])
                or r["throughput_definition"] != entry["definition"]):
            raise ValueError("attempt graph/depth/throughput provenance mismatch")
    predicted = sum(r["status"] == "predicted" for r in rows.values())
    if predicted != 6121 or reasons != {"no_sourced_nominal_throughput": 2645,
                                     "dynamic_control_no_frozen_branch_duration_semantics": 1}:
        raise ValueError("unexpected coverage/unavailable reasons")
    for backend, entry in protocol["entries"].items():
        if entry["source_kind"] == "pinned_fake_backend_configuration":
            asset = output / "inputs" / f"{backend}_configuration.json"
            if sha(asset) != entry["configuration_sha256"]:
                raise ValueError("portable throughput source hash mismatch")
            if json.loads(asset.read_text())[entry["source_field"]] != entry["value"]:
                raise ValueError("portable throughput source field mismatch")
    successful = [r for r in rows.values() if r["status"] == "predicted"]
    y = [float(r["target_seconds"]) for r in successful]
    p = [float(r["prediction"]) for r in successful]
    mae = statistics.fmean(abs(a-b) for a, b in zip(y, p))
    mean = statistics.fmean(y)
    r2 = 1 - math.fsum((a-b)**2 for a, b in zip(y, p)) / math.fsum((a-mean)**2 for a in y)
    stored = read_csv(output / "method_metrics.csv")[0]
    if not (math.isclose(float(stored["mae_seconds"]), mae, rel_tol=1e-12)
            and math.isclose(float(stored["r2_seconds"]), r2, rel_tol=1e-12)):
        raise ValueError("independent metrics mismatch")
    for table in read_csv(output / "same_row_comparison.csv"):
        if int(table["n"]) != predicted or table["row_ids_sha256"] != hashlib.sha256(
                "\n".join(sorted(r["canonical_row_id"] for r in successful)).encode()).hexdigest():
            raise ValueError("comparison successful population mismatch")
    return {"status": "PASS", "assigned": len(rows), "predicted": predicted,
            "unavailable": len(rows)-predicted, "reasons": dict(reasons),
            "mae_seconds": mae, "r2_seconds": r2, "historical_outputs_unchanged": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    print(json.dumps(validate(output) if args.validate else run(output), sort_keys=True))
