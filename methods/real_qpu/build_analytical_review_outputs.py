#!/usr/bin/env python3
"""Build the corrected Scholten output and append the analytical extension to the two-domain reader.

Uses only frozen observations, folds, OOF predictions and the independent
Scholten depth audit. It performs no model fitting, transpilation or execution.
"""
from __future__ import annotations

import ast
import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "benchmark_v1/protocol/analytical_review_handoff.json"
EXT = ROOT / "artifacts/benchmark_v3/real_qpu/analytical_extension"
OLD_READER = ROOT / "artifacts/benchmark_v3/results/two_domain_recovery_reader_v1"
OUT = ROOT / "artifacts/benchmark_v3/results/two_domain_recovery_reader_v2"
CANON = ROOT / "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv"
OUTER = ROOT / "artifacts/benchmark_v2/real_qpu/unified_outer_split_v2.csv"
SEEDS = ROOT / "benchmark_v1/registry/seed_registry.json"
METHOD_ID = "scholten_reported_nominal_throughput_depth_v2"
METHODS = [
    ["compiled_graph_mali_style_v1", "qpu_seven_global_ridge_v1"],
    ["compiled_metadata_mlp_v1", "qpu_seven_global_ridge_v1"],
    ["compiled_common_feature_polynomial_v1", "qpu_seven_global_ridge_v1"],
    ["compiled_graph_mali_style_v1", "hyb_kyoto_composite_r2_gate_time_ridge_v1"],
    ["hyb_nominal_r2_log_cost_ridge_v1", "hyb_kyoto_composite_r2_log_cost_ridge_v1"],
    ["hyb_kyoto_composite_r2_log_cost_ridge_v1", "hyb_kyoto_composite_r2_gate_time_ridge_v1"],
    ["qpu_depth_shots_ridge_v1", "qpu_seven_global_ridge_v1"],
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, data: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for r in data for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        w.writerows(data)


def linear_quantile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (pos - lo) * (xs[hi] - xs[lo])


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None

    def rank(values):
        order = sorted(range(len(values)), key=values.__getitem__)
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and values[order[j]] == values[order[i]]:
                j += 1
            midrank = ((i + 1) + j) / 2
            for k in order[i:j]:
                out[k] = midrank
            i = j
        return out

    rx, ry = rank(x), rank(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = math.fsum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(math.fsum((a - mx) ** 2 for a in rx) * math.fsum((b - my) ** 2 for b in ry))
    return cov / den if den else None


def summarize(valid: list[dict], assigned: int, metric_type: str) -> dict:
    if not valid:
        return {"n": 0, "assigned_n": assigned, "coverage": 0.0, "mae_seconds": "",
                "r2_seconds": "", "spearman_rho": ""}
    actual = [float(r["target_seconds"]) for r in valid]
    pred = [float(r["prediction"]) for r in valid]
    if metric_type == "rank":
        return {"n": len(valid), "assigned_n": assigned, "coverage": len(valid) / assigned,
                "mae_seconds": "", "medae_seconds": "", "log1p_mae": "", "r2_seconds": "",
                "p90_abs_error_seconds": "", "p99_abs_error_seconds": "", "max_abs_error_seconds": "",
                "spearman_rho": spearman(actual, pred)}
    errors = [abs(a - b) for a, b in zip(actual, pred)]
    max_err = max(errors)
    mae = max_err * math.fsum(e / max_err for e in errors) / len(errors) if max_err else 0.0
    mean_y = statistics.fmean(actual)
    residual = [a - b for a, b in zip(actual, pred)]
    centered = [a - mean_y for a in actual]
    rs, cs = max(map(abs, residual)), max(map(abs, centered))
    r2 = ""
    if cs and len(valid) > 1:
        if not rs:
            r2 = 1.0
        else:
            ratio = rs / cs
            sumratio = math.fsum((e / rs) ** 2 for e in residual) / math.fsum((e / cs) ** 2 for e in centered)
            # Multiply the smaller normalized term first: ratio**2 may overflow
            # even where ratio**2 * sumratio is finite.
            penalty = ratio * (ratio * sumratio)
            if math.isfinite(penalty):
                r2 = 1.0 - penalty
    return {"n": len(valid), "assigned_n": assigned, "coverage": len(valid) / assigned,
            "mae_seconds": mae, "medae_seconds": statistics.median(errors),
            "log1p_mae": statistics.fmean(abs(math.log1p(a) - math.log1p(b)) for a, b in zip(actual, pred)),
            "r2_seconds": r2, "r2_null_reason": "" if r2 != "" else "finite_error_or_target_variance_not_representable",
            "p90_abs_error_seconds": linear_quantile(errors, .90),
            "p99_abs_error_seconds": linear_quantile(errors, .99), "max_abs_error_seconds": max_err,
            "spearman_rho": spearman(actual, pred)}


def source_macro_summary(source_metric: dict[str, dict], metric_type: str) -> tuple[object, int, str]:
    supported = [name for name in sorted(source_metric) if source_metric[name]["n"]]
    if metric_type != "seconds":
        return "", len(supported), "rank_only"
    if len(supported) != 3 or len(source_metric) != 3:
        return "", len(supported), "partial_source_support" if supported else "unavailable"
    return statistics.fmean(float(source_metric[s]["mae_seconds"]) for s in supported), 3, "all_three_sources"


def bootstrap_delta(a: list[dict], b: list[dict], handoff_id: str, seed_registry: dict) -> dict:
    amap = {r["canonical_row_id"]: r for r in a if r["status"] == "predicted"}
    bmap = {r["canonical_row_id"]: r for r in b if r["status"] == "predicted"}
    ids = sorted(amap.keys() & bmap.keys())
    if not ids:
        raise ValueError("empty successful pair");
    by_source, group_source = defaultdict(list), {}
    deltas = {}
    for identity in ids:
        src, group = amap[identity]["source_id"], amap[identity]["leakage_group"]
        if group in group_source and group_source[group] != src:
            raise ValueError(f"bootstrap group crosses sources: {group}")
        group_source[group] = src
        deltas[identity] = abs(float(amap[identity]["target_seconds"]) - float(amap[identity]["prediction"])) - abs(
            float(bmap[identity]["target_seconds"]) - float(bmap[identity]["prediction"]))
        by_source[src].append(identity)
    groups = {src: defaultdict(list) for src in by_source}
    for identity in ids:
        groups[amap[identity]["source_id"]][amap[identity]["leakage_group"]].append(identity)
    names = sorted((a[0]["method_id"], b[0]["method_id"]))
    pair_id = "::".join(names)
    protocol = json.loads((ROOT / "benchmark_v1/protocol/analytical_extension.json").read_text())
    seed_protocol = protocol["protocol_id"] if handoff_id == protocol["protocol_id"] else handoff_id
    seed_material = f"{seed_registry['root_seed']}|{seed_registry['streams']['bootstrap']}|{seed_protocol}|{pair_id}|0"
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    pooled_sum = np.zeros(10000, dtype=np.float64)
    pooled_count = np.zeros(10000, dtype=np.float64)
    per_source_replicates = []
    sources = sorted(groups)
    for source in sources:
        source_groups = sorted(groups[source])
        group_sums = np.asarray([math.fsum(deltas[i] for i in groups[source][g]) for g in source_groups])
        group_sizes = np.asarray([len(groups[source][g]) for g in source_groups], dtype=np.float64)
        indices = rng.integers(0, len(source_groups), size=(10000, len(source_groups)))
        source_sum = group_sums[indices].sum(axis=1)
        source_count = group_sizes[indices].sum(axis=1)
        per_source_replicates.append(source_sum / source_count)
        pooled_sum += source_sum
        pooled_count += source_count
    pooled_replicates = pooled_sum / pooled_count
    macro_replicates = np.mean(np.column_stack(per_source_replicates), axis=1)
    pooled_direct = statistics.fmean(deltas.values())
    macro_direct = statistics.fmean(statistics.fmean(deltas[i] for i in by_source[s]) for s in sources)
    return {"method_A": a[0]["method_id"], "method_B": b[0]["method_id"], "n": len(ids),
            "row_set_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "groups_by_source": json.dumps({s: len(groups[s]) for s in sources}, sort_keys=True),
            "seed_uint32": seed, "replicates": 10000,
            "pooled_direct_delta": pooled_direct,
            "pooled_bootstrap_mean": float(np.mean(pooled_replicates)),
            "pooled_ci_95_low": linear_quantile(pooled_replicates, .025),
            "pooled_ci_95_high": linear_quantile(pooled_replicates, .975),
            "source_macro_direct_delta": macro_direct,
            "source_macro_bootstrap_mean": float(np.mean(macro_replicates)),
            "source_macro_ci_95_low": linear_quantile(macro_replicates, .025),
            "source_macro_ci_95_high": linear_quantile(macro_replicates, .975)}


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT,
                        help="new reader directory; must not already exist")
    args = parser.parse_args()
    OUT = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    handoff = json.loads(HANDOFF.read_text())
    protocol = json.loads((ROOT / handoff["frozen_extension"]["protocol_path"]).read_text())
    if sha(ROOT / handoff["scientific_review"]["path"]) != handoff["scientific_review"]["sha256"]:
        raise ValueError("scientific audit hash mismatch")
    if sha(ROOT / handoff["frozen_extension"]["protocol_path"]) != handoff["frozen_extension"]["protocol_sha256"]:
        raise ValueError("frozen protocol hash mismatch")
    if sha(ROOT / handoff["frozen_extension"]["aggregate_receipt_path"]) != handoff["frozen_extension"]["aggregate_receipt_sha256"]:
        raise ValueError("extension receipt hash mismatch")
    audit = json.loads((ROOT / handoff["scientific_review"]["path"]).read_text())
    if audit["scholten_contract_audit"]["affected_predictions"] != 352:
        raise ValueError("Scholten correction support differs from frozen scientific review")
    canonical = {r["canonical_row_id"]: r for r in rows(CANON)}
    outer = {r["canonical_observation_id"]: r for r in rows(OUTER)}
    if len(canonical) != 8767 or set(canonical) != set(outer):
        raise ValueError("canonical/split identity mismatch")
    groups = {i: outer[i]["unified_leakage_group_id"] for i in canonical}
    source = {i: canonical[i]["source_id"] for i in canonical}
    base_ext = rows(EXT / "unified_attempt_envelope.csv")
    ext_by_method = defaultdict(list)
    for r in base_ext:
        ext_by_method[r["method_id"]].append(r)
    if len(ext_by_method) != 11 or any(len(v) != 8767 for v in ext_by_method.values()):
        raise ValueError("frozen analytical extension must have 11 complete row envelopes")
    for m, rr in ext_by_method.items():
        for r in rr:
            i = r["canonical_row_id"]
            c = canonical[i]
            if float(r["target_seconds"]) != float(c["target_seconds"]) or int(r["outer_fold"]) != int(outer[i]["outer_fold"]):
                raise ValueError(f"frozen target/fold mismatch in {m}:{i}")
            r["leakage_group"] = groups[i]

    # Correct only the declared Scholten depth defect. The historical v1 attempts stay intact.
    old = ext_by_method["scholten_reported_nominal_throughput_v1"]
    depth_delta = {d["id"]: d for d in audit["scholten_contract_audit"]["discrepancies"]}
    old_predicted = {r["canonical_row_id"] for r in old if r["status"] == "predicted"}
    if old_predicted != set(depth_delta):
        raise ValueError("v1 Scholten eligible set differs from independent depth audit")
    throughput = protocol["scholten"]["active_entries"]
    corrected = []
    for r in old:
        i = r["canonical_row_id"]
        new = dict(r)
        new["method_id"] = METHOD_ID
        new["method_output_clock"] = "nominal_reported_throughput_seconds"
        new["reader_label"] = "Scholten-style reported-throughput adaptation, corrected quantum-wire depth"
        new["fidelity_class"] = "adaptation"
        new["claim_boundary"] = "Generic compiled quantum-wire depth adaptation; not QV-normalized effective depth or historical CLOPS_v. Corrects v1's omitted measurement layer."
        if r["status"] == "predicted":
            newdepth = depth_delta[i]["contract_depth"]
            new["prediction"] = int(canonical[i]["shots"]) * newdepth / float(throughput[r["backend"]]["CLOPS"])
            new["status"], new["terminal_reason"] = "predicted", ""
            diff = float(new["prediction"]) - float(r["prediction"])
            expected = int(canonical[i]["shots"]) / float(throughput[r["backend"]]["CLOPS"])
            if not math.isclose(diff, expected, rel_tol=1e-12, abs_tol=1e-10):
                raise ValueError(f"Scholten correction is not the contract depth increment: {i}")
        else:
            new["prediction"] = ""
            new["status"], new["terminal_reason"] = r["status"], r["terminal_reason"]
        corrected.append(new)
    ext_by_method[METHOD_ID] = corrected

    core_ids = ["compiled_common_feature_polynomial_v1", "compiled_graph_mali_style_v1", "compiled_metadata_mlp_v1"]
    method_card = {mid: {"method_id": mid, **card}
                   for mid, card in handoff["qpu_comparison"]["reader_method_cards"].items()}
    core_streams = {stream["method_id"]: stream for stream in handoff["qpu_comparison"]["learned_streams"]}
    if set(method_card) != set(core_ids) or set(core_streams) != set(core_ids):
        raise ValueError("handoff must pin exactly three compiled QPU reader cards and OOF streams")
    core_rows = {}
    for mid in core_ids:
        stream = core_streams[mid]
        card = method_card[mid]
        rawpath = ROOT / stream["path"]
        if sha(rawpath) != stream["sha256"]:
            raise ValueError(f"core OOF hash mismatch: {mid}")
        rr = []
        for x in rows(rawpath):
            i = x["canonical_observation_id"]
            y = float(x["actual_seconds"])
            if (y != float(canonical[i]["target_seconds"])
                    or int(x["outer_fold"]) != int(outer[i]["outer_fold"])
                    or x["source_id"] != canonical[i]["source_id"]):
                raise ValueError(f"core target/source/fold mismatch: {mid}:{i}")
            rr.append({"canonical_row_id": i, "source_id": x["source_id"], "backend": canonical[i]["backend"],
                       "outer_fold": int(x["outer_fold"]), "leakage_group": groups[i],
                       "method_id": mid, "evaluation_target_clock": card["evaluation_target_clock"],
                       "method_output_clock": card["method_output_clock"], "target_seconds": y,
                       "prediction": x["predicted_seconds"], "status": x["status"], "terminal_reason": x["terminal_reason"],
                       "fidelity_class": card["fidelity_class"], "reader_label": card["reader_label"],
                       "claim_boundary": card["claim_boundary"], "metric_type": "seconds"})
        if len(rr) != 8767:
            raise ValueError("core OOF does not assign all canonical rows")
        core_rows[mid] = rr

    methods = {m: rr for m, rr in core_rows.items()}
    for method in protocol["methods"]:
        mid = method["method_id"]
        method_card[mid] = method
        for r in ext_by_method[mid]:
            r["reader_label"] = method["reader_label"]
            r["fidelity_class"] = method["fidelity_class"]
            r["claim_boundary"] = method.get("claim_boundary", "See frozen analytical extension method card and paper boundary.")
            r["metric_type"] = "rank" if method["implementation"] == "qcre_rank" else "seconds"
        methods[mid] = ext_by_method[mid]
    method_card[METHOD_ID] = {"method_id": METHOD_ID, "reader_label": "Scholten-style reported-throughput adaptation, corrected quantum-wire depth",
        "fidelity_class": "adaptation", "claim_boundary": handoff["scholten_correction"]["claim_boundary"],
        "method_output_clock": "nominal_reported_throughput_seconds", "metric_type": "seconds",
        "supersedes": "scholten_reported_nominal_throughput_v1 only for contract-compliant presentation"}
    methods[METHOD_ID] = corrected
    if len(methods) != 15:
        raise ValueError("expected three core methods, eleven extension methods and the corrected Scholten route")

    # Exact denominator, common-row and metric audit for every reader row.
    metrics_rows, source_rows, all_attempts = [], [], []
    for mid, rr in methods.items():
        if len(rr) != 8767 or len({r["canonical_row_id"] for r in rr}) != 8767:
            raise ValueError(f"incomplete/duplicate method envelope: {mid}")
        typ = next(r.get("metric_type", "seconds") for r in rr)
        valid = [r for r in rr if r["status"] == "predicted"]
        own = summarize(valid, 8767, typ)
        src_metrics = {}
        for s in sorted({r["source_id"] for r in rr}):
            sr = [r for r in rr if r["source_id"] == s]
            sv = [r for r in sr if r["status"] == "predicted"]
            sm = summarize(sv, len(sr), typ)
            src_metrics[s] = sm
            source_rows.append({"method_id": mid, "source_id": s, **sm,
                "status_counts_json": json.dumps(dict(sorted(Counter(r["status"] for r in sr).items())), sort_keys=True)})
        macro, macro_count, macro_status = source_macro_summary(src_metrics, typ)
        terminal = Counter(r["status"] for r in rr)
        card = method_card[mid]
        metric = {"method_id": mid, "reader_label": card["reader_label"],
            "fidelity_class": card["fidelity_class"], "claim_boundary": card.get("claim_boundary", protocol["method_card_boundary"]),
            "method_output_clock": card["method_output_clock"], "metric_type": typ,
            "evaluation_target_clock": "archived_observed_service_execution_time",
            "assigned_n": 8767, "terminal_counts_json": json.dumps(dict(sorted(terminal.items())), sort_keys=True),
            "source_balanced_macro_mae_seconds": macro,
            "source_macro_observed_sources_n": macro_count, "source_macro_status": macro_status, **own}
        if mid == "scholten_reported_nominal_throughput_v1":
            metric["result_disposition"] = "superseded_implementation_contract_mismatch; retained in attempts for audit"
        elif mid == METHOD_ID:
            metric["result_disposition"] = "corrected_contract_compliant_nominal_adaptation"
        else:
            metric["result_disposition"] = "included_under_declared_method_clock_and_scope"
        metrics_rows.append(metric)
        all_attempts.extend(rr)

    # Keep the two historical extension comparisons and four preregistered new comparisons.
    seed_registry = json.loads(SEEDS.read_text())
    pair_rows, boot_rows = [], []
    old_by_pair = { (r["method_A"], r["method_B"]): r for r in rows(EXT / "paired_bootstrap.csv") }
    for A, B in METHODS:
        aset = {r["canonical_row_id"]: r for r in methods[A] if r["status"] == "predicted"}
        bset = {r["canonical_row_id"]: r for r in methods[B] if r["status"] == "predicted"}
        common = sorted(aset.keys() & bset.keys())
        rowsa, rowsb = [aset[i] for i in common], [bset[i] for i in common]
        aid_hash = hashlib.sha256("\n".join(common).encode()).hexdigest()
        pair_metrics = {}
        for mid, selected in [(A, rowsa), (B, rowsb)]:
            pair_metrics[mid] = summarize(selected, len(common), "seconds")
        pair_rows.append({"method_A": A, "method_B": B, "n_common": len(common), "row_ids_sha256": aid_hash,
            "method_A_metrics_json": json.dumps(pair_metrics[A], sort_keys=True),
            "method_B_metrics_json": json.dumps(pair_metrics[B], sort_keys=True)})
        previous_pair = (A, B) in old_by_pair
        bootstrap_protocol = protocol["protocol_id"] if previous_pair else handoff["handoff_id"]
        boot = bootstrap_delta(rowsa, rowsb, bootstrap_protocol, seed_registry)
        # Historical three intervals are recomputed and checked against the frozen aggregate.
        old = old_by_pair.get((A, B))
        if old:
            oldvals = ast.literal_eval(old["method_A_minus_method_B"])
            macro = oldvals["source_balanced_macro_mae_seconds"]
            if boot["n"] != int(old["common_rows"]) or boot["row_set_sha256"] != old["common_ids_sha256"]:
                raise ValueError(f"preserved extension pair support changed: {A}:{B}")
            if not math.isclose(boot["source_macro_direct_delta"], macro["direct_observed_delta"], rel_tol=1e-11, abs_tol=1e-11):
                raise ValueError(f"preserved extension pair point delta changed: {A}:{B}")
            if not math.isclose(boot["source_macro_ci_95_low"], macro["ci_95_low"], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"preserved extension pair interval changed: {A}:{B}; computed={boot['source_macro_ci_95_low']}; frozen={macro['ci_95_low']}")
        boot_rows.append(boot)

    # Copy historical reader contents, then add this review's transparent extension files.
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite an existing reader: {OUT}")
    shutil.copytree(OLD_READER, OUT)
    qpu = OUT / "qpu"
    write_csv(qpu / "unified_method_attempts.csv", all_attempts)
    write_csv(qpu / "unified_method_metrics.csv", metrics_rows)
    write_csv(qpu / "unified_source_metrics.csv", source_rows)
    write_csv(qpu / "unified_same_row_pairs.csv", pair_rows)
    write_csv(qpu / "unified_paired_bootstrap.csv", boot_rows)

    sim = rows(OLD_READER / "simulator/method_comparison.csv")
    sim_out = qpu.parent / "simulator_method_metrics.csv"
    # A two-domain index view. Retain every context/status/value and the exact source schema.
    write_csv(sim_out, sim)
    sim_sha = sha(OLD_READER / "simulator/method_comparison.csv")
    output_hashes = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            output_hashes[str(p.relative_to(OUT))] = sha(p)
    source_paths = [HANDOFF, ROOT / handoff["scientific_review"]["path"], ROOT / handoff["frozen_extension"]["protocol_path"],
        ROOT / handoff["frozen_extension"]["aggregate_receipt_path"], EXT / "unified_attempt_envelope.csv",
        EXT / "paired_bootstrap.csv", EXT / "graph_instruction_inventory.json", CANON, OUTER, SEEDS,
        ROOT / "benchmark_v1/requirements-verification.txt",
        ROOT / ".gitattributes",
        OLD_READER / "manifest.json", OLD_READER / "simulator/method_comparison.csv"]
    for stream in handoff["qpu_comparison"]["learned_streams"]:
        source_paths.append(ROOT / stream["path"])
    manifest = {"artifact_id": "two-domain-unified-runtime-reader-v2", "schema_version": 2,
        "status": "PASS", "public_release_approved": False, "training_or_measurement_performed": False,
        "assigned_qpu_observations": 8767, "qpu_method_count": len(methods),
        "corrected_scholten": {"method_id": METHOD_ID, "corrected_rows": len(old_predicted),
            "previous_v1_disposition": "implementation_contract_mismatch; retained as historical evidence",
            "equation": "shots * corrected_generic_quantum_wire_depth / reported_nominal_CLOPS"},
        "common_primary_qpu_rows": 8766, "common_sensitivity_rows": 8618,
        "source_balanced_macro_is_primary": True,
        "rebuild_command": "python benchmark_v1/scripts/build_analytical_review_outputs.py --output-dir <new-directory>",
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
        "code_hashes": {p: sha(ROOT / p) for p in [
            "benchmark_v1/scripts/build_analytical_review_outputs.py",
            "benchmark_v1/scripts/validate_analytical_review_outputs.py",
            "benchmark_v1/scripts/audit_analytical_extension.py",
            "benchmark_v1/scripts/build_graph_instruction_inventory.py",
            "benchmark_v1/tests/test_analytical_scientific_review.py",
            "benchmark_v1/tests/test_analytical_review_outputs.py"]},
        "simulator_source_path": "simulator_method_metrics.csv", "simulator_source_sha256": sim_sha,
        "simulator_source_reader": "two_domain_recovery_reader_v1; unchanged method/context rows",
        "historical_reader_v1_modified": False,
        "input_hashes": {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
        "output_hashes": output_hashes,
        "method_ids": sorted(methods),
        "metrics": "Recomputed from saved OOF attempts; no fit or hardware execution",
        "notes": ["Seconds metrics only compare methods on identical successful IDs and compatible output clocks.",
                  "Raw analytical cost and scheduled/rank outputs preserve method_output_clock; they are diagnostics, not mislabeled service time.",
                  "Historical source-local, simulator and companion method classes remain in their own tables and scopes."]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUT / "README.md").write_text(
        "# Two-domain quantum runtime reader v2\n\n"
        "This revision keeps the previous simulator context tables and adds a full-denominator QPU method table for the 8,767 archived observations. "
        "The corrected Scholten method counts its generic compiled quantum-wire depth consistently with the declared local contract. "
        "All old QPU, simulator and historical appendices remain available in their existing subdirectories.\n\n"
        "There is no cross-clock flat leaderboard. Use `qpu/unified_method_metrics.csv` and `qpu/unified_source_metrics.csv` for assigned-envelope coverage and source-balanced scores; "
        "`qpu/unified_same_row_pairs.csv` and `qpu/unified_paired_bootstrap.csv` compare only declared pairs on identical successful identities. "
        "`qpu/unified_method_attempts.csv` retains each method's output and terminal state. `simulator_method_metrics.csv` indexes the unchanged simulator table; "
        "the full simulator configuration, quality and availability tables remain under `simulator/`.\n\n"
        "The methods are local replications/adaptations with explicit claim boundaries, not 15 paper-faithful reproductions. The raw Hyb-HANAS effective cost and QCRE rank keep their native units; "
        "rank has Spearman only. The v1 Scholten calculation remains visible as an implementation-contract mismatch. "
        "A separate graph-hash instruction inventory is linked from the methodology guide.\n\n"
        "Rebuild from the pinned saved inputs into a new directory (the builder never overwrites an existing reader):\n\n"
        "```bash\npython benchmark_v1/scripts/build_analytical_review_outputs.py --output-dir work/rebuilds/two_domain_recovery_reader_v2\n"
        "python benchmark_v1/scripts/validate_analytical_review_outputs.py --reader-dir work/rebuilds/two_domain_recovery_reader_v2\n```\n\n"
        "This is table replay from saved prediction/attempt ledgers. It does not regenerate source circuits, retrain models, or measure a device. "
        "See the repository reproduction guide for input availability, dependency, and clean-clone boundaries.\n")
    report = ["# Reader v2 report", "", "Same-row common QPU results (8,766 IDs; successful predictions only):", "",
              "| Method | Source-balanced MAE (s) | Pooled MAE (s) | Pooled R² |", "| --- | ---: | ---: | ---: |"]
    for mid in ["compiled_graph_mali_style_v1", "compiled_metadata_mlp_v1", "hyb_kyoto_composite_r2_gate_time_ridge_v1",
                "qpu_seven_global_ridge_v1", "qpu_depth_shots_ridge_v1", "hyb_kyoto_composite_r2_log_cost_ridge_v1",
                "compiled_common_feature_polynomial_v1"]:
        m = next(x for x in metrics_rows if x["method_id"] == mid)
        report.append(f"| {mid} | {float(m['source_balanced_macro_mae_seconds']):.4f} | {float(m['mae_seconds']):.4f} | {float(m['r2_seconds']):.4f} |")
    report += ["", "## Full assigned-envelope results", "",
               "Every method keeps all 8,767 assigned identities. Metrics below use only successful predictions; coverage and terminal counts show failures/unavailable rows.", "",
               "| Method | Fidelity | Output clock | Predicted / assigned | Coverage | MAE (s) | R² | Source macro MAE (s) | Disposition |",
               "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for metric in metrics_rows:
        def display(value, digits=4):
            if value == "" or value is None:
                return "—"
            try:
                return f"{float(value):.{digits}g}"
            except (TypeError, ValueError):
                return str(value)
        report.append("| {method_id} | {fidelity_class} | {method_output_clock} | {n} / {assigned_n} | {coverage} | {mae} | {r2} | {macro} | {result_disposition} |".format(
            method_id=metric["method_id"], fidelity_class=metric["fidelity_class"],
            method_output_clock=metric["method_output_clock"], n=metric["n"], assigned_n=metric["assigned_n"],
            coverage=display(metric["coverage"], 3), mae=display(metric["mae_seconds"]),
            r2=display(metric["r2_seconds"], 4), macro=display(metric["source_balanced_macro_mae_seconds"]),
            result_disposition=metric["result_disposition"]))
    report += ["", "Scholten v2 corrects the one-layer depth omission in its 352 supported rows. It predicts only 352/8,767 rows; its error on that restricted support is not comparable to full-coverage methods. The corrected route is a sensitivity adaptation using generic compiled quantum-wire depth, not the paper's QV-normalized effective depth.",
               "Simulator rows preserve their method-specific engine, configuration, clock and quality scope; the methods are not ranked against QPU seconds.",
               "See `manifest.json` for source/output hashes and `qpu/unified_method_metrics.csv` for all methods and coverage."]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    # Refresh manifest hashes for the two generated prose files.
    manifest["output_hashes"] = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    location = str(OUT.relative_to(ROOT)) if OUT.is_relative_to(ROOT) else str(OUT)
    print(json.dumps({"status": "PASS", "reader": location, "methods": len(methods),
        "qpu_rows": len(all_attempts), "corrected_scholten": len(old_predicted),
        "common_primary": len({r["row_ids_sha256"] for r in pair_rows if r["n_common"] == 8766}),
        "paired_comparisons": len(pair_rows), "bootstrap_intervals": len(boot_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
