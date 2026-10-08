#!/usr/bin/env python3
"""Build and verify reader results from the completed qualified QPU experiment.

No model loading, fitting, circuit execution, or changes to frozen evidence.
The six reader files and one validation/index file live beside the saved scores.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_qualified_logical_benchmark as frozen
from mali_full_features import stable_hash
from run_mali_full_features import atomic_bytes

ROOT, OUT = frozen.ROOT, frozen.OUT
SUMMARY = OUT / "summary"
SOURCES = ("mali_real_qpu", "qonductor_single_circuit_ibm", "qpack_mcp")
SOURCE_LABELS = dict(zip(SOURCES, ("Ma–Li", "Qonductor", "QPack MCP")))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def same_number(actual, expected, message):
    if actual == "" or expected == "":
        require(actual == expected, message)
    else:
        require(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10), message)


def measure(values):
    """Independent metric calculation; missing predictions never become zeros."""
    good = [r for r in values if r["status"] == "predicted"]
    y = np.array([float(r["actual_seconds"]) for r in good])
    p = np.array([float(r["predicted_seconds"]) for r in good])
    require(np.isfinite(y).all() and np.isfinite(p).all(), "nonfinite scored value")
    result = dict(assigned_rows=len(values), predicted_rows=len(good),
                  coverage=len(good) / len(values) if values else 0,
                  scored_ids_sha256=frozen.idhash(r["canonical_observation_id"] for r in good),
                  negative_prediction_count=int((p < 0).sum()))
    names = ("mae_seconds", "medae_seconds", "rmse_seconds", "r2_seconds", "log1p_mae",
             "p90_absolute_error_seconds", "p99_absolute_error_seconds", "max_absolute_error_seconds")
    result.update(dict.fromkeys(names, ""))
    if len(good):
        error = np.abs(p - y)
        with np.errstate(over="ignore", invalid="ignore"):
            squared = np.dot(error, error)
            centered = y - y.mean()
            denominator = np.dot(centered, centered)
            r2 = 1 - squared / denominator if denominator > 0 else float("nan")
            metrics = (error.mean(), np.median(error), np.sqrt(squared / len(y)), r2,
                       np.abs(np.log1p(y) - np.log1p(np.maximum(p, 0))).mean(),
                       *np.quantile(error, (.9, .99)), error.max())
        result.update({k: float(v) if math.isfinite(float(v)) else "" for k, v in zip(names, metrics)})
    return result


def check_metrics(published, values):
    for field, value in measure(values).items():
        if field.endswith("sha256"):
            require(published[field] == value, f"metric ID mismatch: {field}")
        else:
            same_number(published[field], value, f"metric mismatch: {field}")


def method_card(method):
    learned = {
        "mali_logical_graph": "Ma–Li-style logical graph + global features",
        "mali_global_mlp": "Matched global-feature MLP",
        "qonductor_budgeted_selector": "Qonductor-style inner-selected regressor family",
        "qonductor_extra_trees": "Qonductor-style Extra Trees",
        "qonductor_random_forest": "Qonductor-style Random Forest",
        "qonductor_gradient_boosting": "Qonductor-style Gradient Boosting",
        "qonductor_adaboost": "Qonductor-style AdaBoost",
        "qonductor_histogram_gradient_boosting": "Qonductor-style Histogram Gradient Boosting",
        "qonductor_polynomial_regression": "Qonductor-style Polynomial Regression",
    }
    if method in learned:
        is_neural = method in frozen.NEURAL
        boundary = ("Original 51-global/178-node schema with fit-only masks; logical reconstruction and nominal-index calibration; three-seed median. No shots/backend ID feature."
                    if is_neural else "Upstream CX/CZ/ECR counter, depth, active width, shots and circuit_count=1; bounded nested search, not the original exhaustive search.")
        return dict(reader_label=learned[method], method_group="learned",
                    fidelity_class="baseline" if method == "mali_global_mlp" else "adaptation",
                    variant="three_seed_median" if is_neural else "inner_selected",
                    claim_boundary=boundary)
    if method == "scholten_original_paper":
        return dict(reader_label="Scholten original-paper eligibility", method_group="analytical",
                    fidelity_class="unavailable", variant="original", claim_boundary="Kernel n and template D needed for original effective depth are not established.")
    if method == "hyb_nominal_r2_log_cost_ridge_v1":
        return dict(reader_label="Hyb-HANAS-style nominal log-cost Ridge", method_group="analytical",
                    fidelity_class="adaptation", variant="inner_selected_log_cost_ridge",
                    claim_boundary="Benchmark-defined train-only Ridge on nominal r2 log-cost components; no full hybrid-NAS reproduction or alternate-snapshot selection.")
    bases = {
        "qcre_source_estimator": ("QCRE source unitary duration", "Pinned QCRE API via BQSKit on nominal compiled/submitted inputs; terminal measurement omitted. Not historical physical timing."),
        "qcre_source_estimator_shot_scaled": ("QCRE source shot-scaled unitary duration", "Pinned QCRE API, terminal measurement omitted; linear shot extension, not measured service overhead."),
        "qiskit_estimate_duration_snapshot": ("Qiskit nominal scheduled duration", "Qiskit estimate_duration on nominal inputs, retaining its measurement convention; not observed service time."),
        "qiskit_shot_scaled_schedule_seconds": ("Qiskit shot-scaled nominal schedule", "Linear shot extension of nominal scheduled duration, including measurement; not measured service overhead."),
        "hyb_nominal_r2_effective_cost_v1": ("Hyb-HANAS-style nominal r2 cost", "Declared one-circuit r2 component adaptation; true-zero survival and exponential overflow are failures, not imputed values."),
        "hyb_nominal_r2_shot_effective_cost_v1": ("Hyb-HANAS-style shot-linear nominal r2 cost", "Declared r2 component and linear shot extension; not the complete hybrid-NAS method."),
        "scholten_unified_nominal_throughput": ("Scholten-style nominal ordinary-depth throughput", "Source-backed nominal throughput with ordinary compiled wire depth; not established original QV effective depth or historical job calibration."),
    }
    base, separator, variant = method.partition("__")
    require(separator and base in bases and variant in ("raw", "affine", "log_affine"), f"missing method card: {method}")
    label, boundary = bases[base]
    suffix = {"raw": "raw", "affine": "outer-train affine", "log_affine": "outer-train log-affine"}[variant]
    if variant != "raw":
        boundary += " Calibration is an added benchmark model fitted only on successful outer-train rows; its intercept is not an isolated physical launch overhead."
    return dict(reader_label=f"{label} — {suffix}", method_group="analytical",
                fidelity_class="analytical_proxy" if variant == "raw" else "adaptation",
                variant=variant, claim_boundary=boundary)


def quantiles(values):
    a = np.asarray(values, dtype=float)
    require(len(a) and np.isfinite(a).all(), "invalid profile values")
    return dict(zip(("min", "p25", "median", "p75", "p90", "max"),
                    (float(x) for x in np.quantile(a, (0, .25, .5, .75, .9, 1)))))


def profile(context, transforms):
    panel = context["panel"]
    compiled = frozen.indexed(frozen.rows(frozen.REP / "representation_rows.csv"), "canonical_row_id")
    result = dict(rows=len(panel), groups=len({r["group_id"] for r in panel}),
                  source_counts=dict(Counter(r["source_id"] for r in panel)),
                  missing_global_features=sum(not math.isfinite(float(r[f"g{i:02d}"])) for r in panel for i in range(51)),
                  fold_transforms=transforms, source_profiles={},
                  qualification="reconstruction-qualified; not all exact historical logical instances",
                  exclusions={"qonductor_unverified_recipe_candidates": 2835,
                              "qonductor_without_qualified_logical_extraction": 1417},
                  notes=["A repeated observation may change backend, shots or measurement occasion; repeated circuit hashes are not automatically duplicate labels.",
                         "Named gate count sums only the 44 encoded operation names, not measurement/barrier or all operations.",
                         "Missing analytical predictions are not zero-filled. Encoder padding zeros are not measured zero T1/T2.",
                         "Source boundaries and nominal snapshots differ; pooled correlations are not causal hardware effects."])
    for source in SOURCES:
        selected = [r for r in panel if r["source_id"] == source]
        stats = {}
        for name, get in (
            ("runtime_seconds", lambda r: r["target_seconds"]),
            ("shots", lambda r: r["shots"]),
            ("logical_allocated_width", lambda r: r["g44"]),
            ("logical_depth", lambda r: r["g45"]),
            ("named_logical_gate_count", lambda r: sum(float(r[f"g{i:02d}"]) for i in range(44))),
            ("compiled_depth", lambda r: compiled[r["canonical_observation_id"]]["structural_depth"]),
            ("compiled_two_qubit_count", lambda r: compiled[r["canonical_observation_id"]]["two_qubit_count"]),
        ):
            stats[name] = quantiles([float(get(r)) for r in selected])
        result["source_profiles"][source] = dict(rows=len(selected), row_fraction=len(selected)/len(panel),
            groups=len({r["group_id"] for r in selected}),
            distinct_logical_input_digests=len({r["logical_input_digest"] for r in selected}),
            distinct_source_qasm_hashes=len({r["source_qasm_sha256"] for r in selected if r["source_qasm_sha256"]}),
            distinct_compiled_qasm_hashes=len({r["compiled_qasm_sha256"] for r in selected if r["compiled_qasm_sha256"]}),
            backend_counts=dict(sorted(Counter(r["backend"] for r in selected).items())),
            shot_counts=dict(sorted(Counter(r["shots"] for r in selected).items())), distributions=stats)
    return result


def verify_evidence(context):
    """Check saved evidence without invoking validators that rewrite old files."""
    inputs, transforms, seed_rows = {}, [], []

    def pin(path):
        inputs[str(Path(path).relative_to(ROOT))] = frozen.sha(path)

    def verify_manifest(path, relative_outputs="output_hashes"):
        receipt = json.loads(path.read_text())
        require(receipt["status"] == "complete" and receipt["pins"] == context["pins"], f"incomplete/drift: {path}")
        pin(path)
        for name, expected in receipt[relative_outputs].items():
            require(frozen.sha(path.parent / name) == expected, f"hash drift: {path.parent / name}")
            pin(path.parent / name)
        return receipt

    verify_manifest(SUMMARY / "manifest.json")
    verify_manifest(OUT / "analytical/manifest.json")
    for path in (OUT / "manifest.json", frozen.CONTRACT, Path(frozen.__file__),
                 OUT / "panel.csv", OUT / "outer_splits.csv", OUT / "inner_splits.csv",
                 OUT / "candidates.json", frozen.NATIVE, frozen.COMPONENTS,
                 frozen.REP / "representation_rows.csv", OUT / "analytical/qcre_source_receipt.json",
                 OUT / "analytical/qcre_context_cache.json", OUT / "analytical/execution_receipt.json",
                 Path(__file__), ROOT / "benchmark_v1/scripts/resume_qualified_analytical.py"):
        pin(path)
    # Read-only reuse preflight verifies source revision, compiled bytes and
    # snapshot assets. It does not invoke QCRE or write an execution log.
    import resume_qualified_analytical as reuse
    reps, _, prior_cache, reused_ids, reuse_pins = reuse.prepare(context)
    qcre_receipt = json.loads((OUT / "analytical/qcre_source_receipt.json").read_text())
    cache = json.loads((OUT / "analytical/qcre_context_cache.json").read_text())
    execution = json.loads((OUT / "analytical/execution_receipt.json").read_text())
    require(execution["status"] == "complete" and execution["pins"] == reuse_pins
            and qcre_receipt["pins"] == reuse_pins and cache["pins"] == reuse_pins, "QCRE execution pins mismatch")
    require(execution["fresh_outer_train_calibration"] and not execution["previous_cohort_fitted_models_reused"], "analytical old-fit reuse")
    for name, path in (("analytical_manifest", OUT / "analytical/manifest.json"),
                       ("summary_manifest", SUMMARY / "manifest.json"),
                       ("qcre_source_receipt.json", OUT / "analytical/qcre_source_receipt.json")):
        require(execution["output_hashes"][name] == frozen.sha(path), "analytical receipt link drift")
    require(qcre_receipt["cache_sha256"] == frozen.sha(OUT / "analytical/qcre_context_cache.json"), "QCRE cache drift")
    require(len(cache["contexts"]) == 374 and len(prior_cache) == 300 and len(reused_ids) == 340, "QCRE context count drift")
    for key, value in prior_cache.items():
        require(cache["contexts"][key] == value, "QCRE reused raw context mismatch")
    for path in (reuse.STRICT / "raw_attempts.csv", reuse.STRICT / "run_manifest.json",
                 reuse.STRICT / "adapter_preflight.json", reuse.REGISTRY,
                 ROOT / "benchmark_v1/registry/CURRENT.json"):
        pin(path)
    attempts = frozen.rows(SUMMARY / "oof_predictions.csv")
    by_method = defaultdict(dict)
    for row in attempts:
        rid = row["canonical_observation_id"]
        require(rid in context["ledger"], "unknown scored row")
        src = context["ledger"][rid]
        for key in ("source_id", "backend", "logical_input_tier", "group_id"):
            require(row[key] == src[key], f"prediction context mismatch: {key}")
        require(int(row["outer_fold"]) == int(context["outer"][rid]["outer_fold"]), "OOF fold mismatch")
        require(float(row["actual_seconds"]) == context["targets"][rid], "target mismatch")
        require(row["evaluation_target_clock"] == "archived_observed_service_execution_time", "target clock mismatch")
        require(rid not in by_method[row["method_id"]], "duplicate method-row")
        if row["status"] == "predicted":
            require(math.isfinite(float(row["predicted_seconds"])), "nonfinite prediction")
        else:
            require(row["predicted_seconds"] == "" and row["terminal_reason"], "failure imputed or unexplained")
        by_method[row["method_id"]][rid] = row
    require(len(by_method) == 32 and len(attempts) == 32 * 4515, "attempt envelope mismatch")
    require(all(set(v) == set(context["ledger"]) for v in by_method.values()), "method denominator mismatch")
    for rid, source in context["ledger"].items():
        value = cache["contexts"][reuse.context_key(source, reps[rid])]["value"]
        same_number(by_method["qcre_source_estimator__raw"][rid]["predicted_seconds"], value, "QCRE raw cache join mismatch")
        same_number(by_method["qcre_source_estimator_shot_scaled__raw"][rid]["predicted_seconds"],
                    value * float(source["shots"]), "QCRE shot scaling mismatch")
    for fold in range(5):
        parts = frozen.partitions(context, fold)
        transform_path = OUT / f"neural/fold_{fold}/transform.json"
        tfm = json.loads(transform_path.read_text())
        pin(transform_path)
        for branch in ("global", "node"):
            require(tfm[branch]["fit_ids_sha256"] == stable_hash(parts["fit"]), "transform fit IDs mismatch")
        globals_fit = np.array([[float(context["ledger"][i][f"g{j:02d}"]) for j in range(51)] for i in parts["fit"]])
        retained = np.flatnonzero(globals_fit.sum(axis=0) > 0)
        require(retained.tolist() == tfm["global"]["retained_indices"], "global feature mask mismatch")
        require(np.allclose(globals_fit[:, retained].mean(axis=0), tfm["global"]["mean"]), "global mean mismatch")
        require(np.allclose(globals_fit[:, retained].std(axis=0, ddof=1), tfm["global"]["std"]), "global std mismatch")
        require((globals_fit[:, retained].std(axis=0, ddof=1) > 1e-6).tolist() == tfm["global"]["variable_mask"], "global variable mask mismatch")
        transforms.append(dict(fold=fold, raw_globals=51, retained_globals=len(retained),
                               variable_globals=sum(tfm["global"]["variable_mask"]),
                               raw_node_fields=178, variable_node_fields=sum(tfm["node"]["variable_mask"]),
                               fit_rows=len(parts["fit"]), validation_rows=len(parts["validation"]), test_rows=len(parts["test"])))
        for method in frozen.NEURAL:
            cells = []
            for seed in frozen.SEEDS:
                folder = OUT / f"neural/fold_{fold}/{method}/seed_{seed}"
                receipt = json.loads((folder / "manifest.json").read_text())
                identity = receipt["identity"]
                require(receipt["status"] == "complete" and receipt["epochs"] == 500 and 0 <= receipt["best_epoch"] < 500, "neural completion mismatch")
                for key, expected in {**context["pins"], "method": method, "fold": fold, "seed": seed, "transform": stable_hash(tfm)}.items():
                    require(identity[key] == expected, f"neural identity drift: {key}")
                require(identity["cuda"] and "+cu" in identity["torch"], "neural CUDA evidence missing")
                for key in ("fit", "validation", "test"):
                    require(identity[key + "_ids"] == frozen.idhash(parts[key]), "neural partition drift")
                pin(folder / "manifest.json")
                for name, expected in receipt["outputs"].items():
                    require(frozen.sha(folder / name) == expected, f"neural output drift: {name}")
                    pin(folder / name)
                predictions = frozen.indexed(frozen.rows(folder / "predictions.csv"))
                require(set(predictions) == set(parts["test"]), "neural test membership mismatch")
                for rid, row in predictions.items():
                    require(int(row["seed"]) == seed and float(row["actual_seconds"]) == context["targets"][rid], "neural seed/target mismatch")
                    seed_rows.append({**context["ledger"][rid], **row, "method_id": method, "seed": seed, "status": "predicted"})
                cells.append(predictions)
            for rid in parts["test"]:
                median = np.median([float(c[rid]["predicted_seconds"]) for c in cells])
                same_number(by_method[method][rid]["predicted_seconds"], median, "three-seed median mismatch")
        receipt_path = OUT / f"regression/fold_{fold}.json"
        receipt = json.loads(receipt_path.read_text())
        predpath = OUT / f"regression/fold_{fold}.csv"
        require(receipt["status"] == "complete" and receipt["pins"] == context["pins"], "regression completion mismatch")
        require(frozen.sha(predpath) == receipt["predictions_sha256"], "regression prediction hash mismatch")
        pin(receipt_path); pin(predpath)
        for row in frozen.rows(predpath):
            require(row == by_method[row["method_id"]][row["canonical_observation_id"]], "regression summary mismatch")
        for family, chosen in receipt["selections"].items():
            scores_path = OUT / f"regression/fold_{fold}_{family.lower().replace(' ', '_')}.json"
            scores = json.loads(scores_path.read_text())
            require(scores["pins"] == context["pins"], "regression search pin mismatch")
            good = [r for r in scores["scores"] if r["status"] == "scored" and math.isfinite(float(r["mean_inner_r2"]))]
            require(chosen == max(good, key=lambda r: (float(r["mean_inner_r2"]), -int(r["candidate_index"]))), "candidate selection mismatch")
            pin(scores_path)
        families = json.loads(frozen.CONTRACT.read_text())["regression"]["families"]
        winner = max(families, key=lambda f: (float(receipt["selections"][f]["mean_inner_r2"]), -families.index(f)))
        require(winner == receipt["selected_family"], "regressor family selection mismatch")
        winner_id = "qonductor_" + winner.lower().replace(" ", "_")
        for rid in parts["test"]:
            same_number(by_method[winner_id][rid]["predicted_seconds"], by_method["qonductor_budgeted_selector"][rid]["predicted_seconds"], "selector prediction mismatch")
    for row in frozen.rows(OUT / "analytical/attempts.csv"):
        require(row == by_method[row["method_id"]][row["canonical_observation_id"]], "analytical summary mismatch")
    fits = frozen.rows(OUT / "analytical/calibration_fits.csv")
    require(len(fits) == 105, "analytical fit receipt count mismatch")
    for fit in fits:
        values = by_method[fit["method_id"] + "__raw"]
        eligible = [i for i in frozen.partitions(context, int(fit["outer_fold"]))["train"] if values[i]["status"] == "predicted"]
        require(fit["fit_id_sha256"] == frozen.idhash(eligible) and int(fit["fit_rows"]) == len(eligible), "analytical fit includes wrong rows")
        if fit["variant"] != "raw":
            variant = by_method[fit["method_id"] + "__" + fit["variant"]]
            for rid in frozen.partitions(context, int(fit["outer_fold"]))["test"]:
                raw = values[rid]
                if raw["status"] != "predicted":
                    require(variant[rid]["status"] == raw["status"], "analytical failure disappeared after calibration")
                    continue
                x = float(raw["predicted_seconds"])
                z = float(fit["slope"]) * (math.log1p(x) if fit["variant"] == "log_affine" else x) + float(fit["intercept"])
                try:
                    expected = max(0., math.expm1(z) if fit["variant"] == "log_affine" else z)
                except OverflowError:
                    require(variant[rid]["status"] == "overflow", "calibration overflow lost")
                    continue
                same_number(variant[rid]["predicted_seconds"], expected, "saved calibration equation mismatch")
    from run_common_panel_hyb import digest_ids, feature
    components = frozen.indexed(frozen.rows(frozen.COMPONENTS), "canonical_row_id")
    for fold in range(5):
        path = OUT / f"analytical/hyb_ridge_fold_{fold}.json"
        receipt = json.loads(path.read_text()); pin(path)
        require(receipt["pins"] == context["pins"], "Hyb Ridge pins mismatch")
        eligible = sorted(i for i in frozen.partitions(context, fold)["train"] if feature("hyb_nominal_r2_log_cost_ridge_v1", components[i]) is not None)
        fit = receipt["fits"][0]
        require(fit["fit_ids_sha256"] == digest_ids(eligible) and fit["eligible_fit_rows"] == len(eligible), "Hyb Ridge fit membership mismatch")
    return by_method, seed_rows, transforms, inputs


def reader_tables(context, by_method, seed_rows):
    published = frozen.rows(SUMMARY / "method_metrics.csv")
    comparison = []
    for row in published:
        method = row["method_id"]
        values = list(by_method[method].values())
        check_metrics(row, values)
        source_scores = [measure([r for r in values if r["source_id"] == s])["mae_seconds"] for s in SOURCES]
        macro = float(np.mean(source_scores)) if all(v != "" for v in source_scores) else ""
        same_number(row["source_balanced_mae_seconds"], macro, "source-balanced score mismatch")
        clocks = {r["method_output_clock"] for r in values}
        require(len(clocks) == 1, "mixed method output clocks")
        comparison.append({"method_id": method, **method_card(method),
            "evaluation_target_clock": "archived_observed_service_execution_time",
            "method_output_clock": next(iter(clocks)), **{k: v for k, v in row.items() if k != "method_id"}})
    for row in frozen.rows(SUMMARY / "stratified_metrics.csv"):
        values = [r for r in by_method[row["method_id"]].values() if r[row["slice_type"]] == row["slice_value"]]
        check_metrics(row, values)
    shared = []
    for row in frozen.rows(SUMMARY / "paired_comparisons.csv"):
        left, right = by_method[row["left_method"]], by_method[row["right_method"]]
        ids = sorted(i for i in left if left[i]["status"] == right[i]["status"] == "predicted")
        a, b = measure([left[i] for i in ids]), measure([right[i] for i in ids])
        require(int(row["n"]) == len(ids) and row["ids_sha256"] == frozen.idhash(ids), "paired row mismatch")
        require(int(row["groups"]) == len({context["ledger"][i]["group_id"] for i in ids}), "paired group mismatch")
        same_number(row["observed_mae_delta_seconds"], a["mae_seconds"] - b["mae_seconds"], "paired point estimate mismatch")
        require(int(row["bootstrap_seed"]) == 42 and int(row["bootstrap_replicates"]) == 10000, "bootstrap settings drift")
        require(float(row["bootstrap_ci_low_seconds"]) <= float(row["bootstrap_ci_high_seconds"]), "invalid interval")
        shared.append({**row, "left_reader_label": method_card(row["left_method"])["reader_label"],
            "right_reader_label": method_card(row["right_method"])["reader_label"],
            "left_mae_seconds_on_shared_rows": a["mae_seconds"], "right_mae_seconds_on_shared_rows": b["mae_seconds"],
            "left_r2_on_shared_rows": a["r2_seconds"], "right_r2_on_shared_rows": b["r2_seconds"]})
    require(len(shared) == 465, "paired comparison count mismatch")
    # Independently reproduce the two intervals used in the main conclusion.
    # Other intervals retain the frozen, hash-pinned calculation as their source.
    for row in shared:
        pair = {row["left_method"], row["right_method"]}
        if pair not in ({"mali_global_mlp", "mali_logical_graph"},
                        {"mali_logical_graph", "qonductor_budgeted_selector"}):
            continue
        differences = defaultdict(list)
        left, right = by_method[row["left_method"]], by_method[row["right_method"]]
        for rid in sorted(left):
            y = context["targets"][rid]
            differences[context["ledger"][rid]["group_id"]].append(
                abs(y-float(left[rid]["predicted_seconds"])) - abs(y-float(right[rid]["predicted_seconds"])))
        sums = np.array([sum(v) for _,v in sorted(differences.items())])
        counts = np.array([len(v) for _,v in sorted(differences.items())])
        choices = np.random.default_rng(42).integers(len(sums), size=(10000,len(sums)))
        estimates = sums[choices].sum(axis=1) / counts[choices].sum(axis=1)
        for key, value in (("bootstrap_mean_delta_seconds", estimates.mean()),
                           ("bootstrap_ci_low_seconds", np.quantile(estimates,.025)),
                           ("bootstrap_ci_high_seconds", np.quantile(estimates,.975))):
            same_number(row[key], value, "main grouped-bootstrap interval mismatch")
    seed_metrics = []
    for method in frozen.NEURAL:
        for seed in frozen.SEEDS:
            values = [r for r in seed_rows if r["method_id"] == method and r["seed"] == seed]
            require(len(values) == 4515, "seed OOF denominator mismatch")
            for source in ("all_sources", *SOURCES):
                selected = values if source == "all_sources" else [r for r in values if r["source_id"] == source]
                seed_metrics.append(dict(method_id=method, seed=seed, source_slice=source, **measure(selected)))
    failures = Counter((method, r["status"], r["terminal_reason"], r["source_id"], r["backend"])
                       for method, values in by_method.items() for r in values.values() if r["status"] != "predicted")
    failure_rows = [dict(method_id=m, status=s, reason=reason, source_id=source, backend=backend, rows=count)
                    for (m, s, reason, source, backend), count in sorted(failures.items())]
    return comparison, shared, seed_metrics, failure_rows


def markdown(comparison, shared, data):
    by_id = {r["method_id"]: r for r in comparison}
    lines = ["# Qualified real-QPU benchmark results", "", "Status: completed and independently checked against saved evidence.", "",
        "This comparison assigns the same 4,515 archived observations to every method under fresh grouped outer folds. It contains 32 variants, not 32 paper methods. Simulator results remain a separate engine/clock comparison; no simulator measurement was changed by this run.", "",
        "## Data and protocol", "", "| Source | Included observations | Conservative groups | Input qualification |", "| --- | ---: | ---: | --- |"]
    qualifications = ("Exact source logical QASM; historical compiled layout not recovered", "Archive-supported MQT recipe; exact submitted physical QASM, not proven original logical DAG", "Six source-backed QAOA structures; historical optimizer angles and submitted routing absent")
    for source, qualification in zip(SOURCES, qualifications):
        p = data["source_profiles"][source]
        lines.append(f"| {SOURCE_LABELS[source]} | {p['rows']:,} | {p['groups']} | {qualification} |")
    lines += ["", "The input filter is independent of runtime and prediction error: Ma–Li 340 is retained, Qonductor 4,482 is reduced to 230 by archive-supported recipe eligibility, and QPack MCP 3,945 retains its explicit structural qualification. The 4,252 excluded Qonductor rows comprise 2,835 unverified recipe candidates and 1,417 rows without qualified logical extraction. Reconstruction supplies inputs, never new runtime labels.", "",
        "All methods share 165 conservative circuit/workflow/model-input groups and five outer/four inner folds. Neural fits use inner folds 1–3 and inner fold 0 for epoch selection; regressors tune across all four inner folds and refit outer-train. Preprocessing is fitted only inside the corresponding training partition. These are not backend-held-out or family-held-out splits.", "",
        "Graph and matched MLP use the upstream 51-global/178-node schema, three CUDA seeds (42, 1234, 31415), 500 epochs per cell and per-observation median predictions. The saved fit-only mask retains 40 globals in each fold; it is not an ad hoc seven-feature model. They do not receive shots or backend/source IDs. Qonductor-style regressors preserve the upstream CX/CZ/ECR counter named `swap`, depth, active width, shots and circuit_count=1; their bounded inner search is disclosed rather than called exhaustive reproduction.", "",
        "The evaluation target is archived provider execution/service time. Ma–Li averages three `Result.time_taken` measurements at 1,024 shots; Qonductor supplies one-circuit job labels with recorded shots; QPack supplies one-circuit duration in milliseconds converted once to seconds. Nominal snapshots match backend names, not proven job-day calibration. Method output clocks are separate columns in the result table.", "", "## Learned methods on the same 4,515 test observations", "",
        "| Method | Coverage | MAE (s) | R² | Source-balanced MAE (s) | p99 error (s) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    ordered = ["mali_logical_graph", "mali_global_mlp", "qonductor_budgeted_selector", "qonductor_extra_trees", "qonductor_random_forest", "qonductor_gradient_boosting", "qonductor_adaboost", "qonductor_histogram_gradient_boosting", "qonductor_polynomial_regression"]
    for method in ordered:
        r = by_id[method]
        lines.append(f"| {r['reader_label']} | 100% | {float(r['mae_seconds']):.3f} | {float(r['r2_seconds']):.3f} | {float(r['source_balanced_mae_seconds']):.3f} | {float(r['p99_absolute_error_seconds']):.3f} |")
    lines += ["", "## Analytical variants and coverage", "", "Raw scheduled duration, effective cost and throughput estimates are evaluated against observed service time but keep their original meaning. Affine/log-affine calibration is an added model fitted on successful outer-train rows only; it is not a claim about a physical launch-overhead coefficient. All raw and calibrated variants are retained, with no test-driven choice of the best variant.", "",
        "| Method family | Successful / assigned | What remains unavailable |", "| --- | ---: | --- |",
        "| QCRE source unitary duration | 4,515 / 4,515 | None; terminal measurement is explicitly omitted |",
        "| Qiskit nominal schedule | 4,515 / 4,515 | None; measurement follows Qiskit's duration convention |",
        "| Hyb-HANAS-style raw r2 cost | 4,175 / 4,515 | 148 Kyoto zero-survival cases and 192 Osaka exponential overflows |",
        "| Hyb-HANAS-style log-cost Ridge | 4,367 / 4,515 | 148 Kyoto zero-survival cases; log representation can retain the 192 Osaka cases |",
        "| Scholten-style nominal ordinary-depth throughput | 3,399 / 4,515 | 1,116 rows lack eligible throughput input; snapshot gate data does not establish CLOPS |",
        "| Scholten original paper | 0 / 4,515 | Original kernel n / template D required for effective depth are not established |", "",
        "Missing values and failures are never imputed as zero. Polynomial regression has three negative predictions, retained in raw-seconds metrics; only the declared log-error calculation projects them to zero. This is a limitation, not a post-hoc reason to repair predictions.", "", "## Findings and limits", ""]
    for left, right, text in (("mali_global_mlp", "mali_logical_graph", "MLP minus graph"), ("mali_logical_graph", "qonductor_budgeted_selector", "Graph minus inner-selected regressor")):
        row = next(r for r in shared if {r["left_method"], r["right_method"]} == {left, right})
        sign = 1 if row["left_method"] == left else -1
        lo, hi = sorted(sign*float(row[k]) for k in ("bootstrap_ci_low_seconds", "bootstrap_ci_high_seconds"))
        lines.append(f"- {text} MAE: {sign*float(row['observed_mae_delta_seconds']):+.3f} s; 95% group-bootstrap interval [{lo:+.3f}, {hi:+.3f}] s. The interval includes zero.")
    lines += ["- Graph has the lowest pooled learned-method MAE point estimate, but these intervals do not establish a robust advantage over the matched MLP or inner-selected regressors. The selector has slightly lower source-balanced MAE.",
        "- QPack accounts for 87.4% of observations but only six conservative groups. Repeated contexts do not provide 3,945 independent circuit structures. Report pooled and source-balanced results together.",
        "- Lower partial-coverage MAE is not a full-cohort win. Use each method pair's identical successful test rows; the pair table reports both MAEs/R² and its row/group count.",
        "- Scheduled timing and observed service labels have different boundaries. Calibration improvement shows a learnable mapping in this cohort, not equivalence of the clocks or validation of historical snapshot accuracy.",
        "- This run does not isolate individual feature effects, prove historical logical-instance recovery for reconstructed inputs, or establish backend/family OOD performance. Such claims require separate evidence.", "", "## Files and reproduction", "",
        "- [method_comparison.csv](method_comparison.csv): all 32 variants, fidelity/claim boundary, separate clocks, coverage and error metrics.",
        "- [shared_row_comparisons.csv](shared_row_comparisons.csv): all 465 pairs, both methods' scores on identical successful test rows, observed differences and saved grouped-bootstrap intervals.",
        "- [dataset_profile.json](dataset_profile.json): per-source width/depth/gates/shots/runtime distributions, reconstruction counts and saved feature masks.",
        "- [seed_metrics.csv](seed_metrics.csv): every neural seed, pooled and by source; no best-seed selection.",
        "- [failure_counts.csv](failure_counts.csv): method/status/reason/source/backend counts.",
        "- [final_validation.json](final_validation.json): evidence and reader-file hashes, completed-cell and independent metric checks.", "",
        "The original `oof_predictions.csv`, `method_metrics.csv`, `stratified_metrics.csv`, `paired_comparisons.csv` and `manifest.json` remain unchanged. This reader is a reporting supplement, not a new experiment.", "",
        "From the repository root, with the prepared source inputs and frozen evidence available:", "", "```bash",
        "python benchmark_v1/scripts/finalize_qualified_benchmark.py --stage build",
        "python benchmark_v1/scripts/finalize_qualified_benchmark.py --stage validate", "```", "",
        "These commands require NumPy and the documented local evidence; they do not train, infer or execute circuits. A report replay is not a clean-clone source refit. Third-party QPU observations and row-level derivatives are excluded from the reduced public distribution; aggregate tables and processing documentation can be staged separately on explicit publication instruction.", ""]
    return "\n".join(lines).encode()


def csv_bytes(values):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(values)
    return stream.getvalue().encode()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def finalize(stage):
    context = frozen.load_context()
    by_method, seed_rows, transforms, input_hashes = verify_evidence(context)
    comparison, shared, seeds, failures = reader_tables(context, by_method, seed_rows)
    data = profile(context, transforms)
    outputs = {"method_comparison.csv": csv_bytes(comparison),
               "shared_row_comparisons.csv": csv_bytes(shared),
               "seed_metrics.csv": csv_bytes(seeds), "failure_counts.csv": csv_bytes(failures),
               "dataset_profile.json": json_bytes(data), "README.md": markdown(comparison, shared, data)}
    import hashlib
    result = dict(status="PASS", scope="completed qualified real-QPU reporting; simulator unchanged",
        pins=context["pins"], assigned_rows=4515, groups=165, method_variants=len(comparison),
        oof_attempt_rows=32*4515, neural_cells=30, regression_outer_folds=5,
        calibration_fit_receipts=105, hyb_ridge_fit_receipts=5, paired_comparisons=len(shared),
        seed_metric_rows=len(seeds), input_hashes=input_hashes,
        report_hashes={name:hashlib.sha256(value).hexdigest() for name,value in outputs.items()},
        checks=["frozen input/code/graph/split hashes", "group leakage and fresh partition support",
                "30 completed CUDA cell/checkpoint/prediction identities", "fit-only global masks/normalization",
                "three-seed median reproduction", "regression inner-selected candidates/family and saved predictions",
                "105 analytical outer-train fit ID receipts and saved calibration equations", "5 Hyb Ridge eligible outer-train receipts",
                "QCRE source/snapshot/cache lineage and raw/shot scaling",
                "32 complete method attempt envelopes and target/context joins", "independent pooled/source/backend/tier metrics",
                "465 identical-success-row pair identities and observed MAE differences",
                "independent reproduction of graph/MLP and graph/selector bootstrap intervals"],
        bootstrap_interval_policy="Reuse hash-pinned 10000-replicate grouped intervals; do not treat bootstrap mean as the observed point estimate.",
        limitations=["Reconstruction-qualified, not all exact historical logical instances",
                     "Nominal, not proven job-day snapshots", "Method-specific inputs and archived source clock boundaries differ",
                     "Partial analytical coverage and few QPack structural groups", "No new feature ablation, simulator run or public release"])
    outputs["final_validation.json"] = json_bytes(result)
    for name, value in outputs.items():
        path = SUMMARY / name
        if stage == "build":
            atomic_bytes(path, value)
        else:
            require(path.read_bytes() == value, f"reader artifact drift: {name}")
    require(all(frozen.sha(ROOT / name) == expected for name,expected in input_hashes.items()), "input modified during reporting")
    return {k: result[k] for k in ("status", "assigned_rows", "groups", "method_variants", "neural_cells", "paired_comparisons")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("build", "validate"), default="validate")
    args = parser.parse_args()
    print(json.dumps(finalize(args.stage), sort_keys=True))


if __name__ == "__main__":
    main()
