#!/usr/bin/env python3
"""Recompute published simulator scores from saved labels and OOF predictions.

Standard library only. Does not import runners, fit models, run inference,
execute simulators, download inputs or write files. Inventory verification is
a separate check. QPU scores cannot be recomputed from this reduced checkout.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 1234, 31415)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def number(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"invalid nonnegative measurement/prediction: {value}")
    return result


def equal(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{label}: computed={actual}, recorded={expected}")


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def metrics(pairs):
    if not pairs:
        raise ValueError("no evaluable prediction/target pairs")
    actual = [number(a) for a, _ in pairs]
    predicted = [number(p) for _, p in pairs]
    errors = [abs(a - p) for a, p in zip(actual, predicted)]
    total = math.fsum((a - statistics.mean(actual)) ** 2 for a in actual)
    squared_error = math.fsum((a - p) ** 2 for a, p in zip(actual, predicted))
    return {
        "metric_n": len(pairs),
        "mae_seconds": statistics.mean(errors),
        "medae_seconds": statistics.median(errors),
        "log1p_mae_seconds": statistics.mean(
            abs(math.log1p(a) - math.log1p(p)) for a, p in zip(actual, predicted)),
        "r2_seconds": 1 - squared_error / total if len(pairs) >= 2 and total else None,
        "p90_absolute_error_seconds": percentile(errors, 0.9),
        "p99_absolute_error_seconds": percentile(errors, 0.99),
        "max_absolute_error_seconds": max(errors),
    }


def unique_rows(rows, fields):
    output = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in output:
            raise ValueError(f"duplicate identity: {key}")
        output[key] = row
    return output


def seed_median(row, output, columns):
    values = [number(row[column]) for column in columns]
    equal(statistics.median(values), number(row[output]), output)


def collapse_aliases(rows):
    """One score per exact hash; aliases must repeat the same target/prediction."""
    unique_rows(rows, ("panel_member_id",))
    output = {}
    for row in rows:
        key = row["source_sha256"]
        value = (row["fold"], row["status"], row["target_seconds"], row["prediction_seconds"])
        if key in output and value != output[key]:
            raise ValueError(f"inconsistent exact-hash aliases: {key}")
        output[key] = value
    return [(number(a), number(p)) for _, status, a, p in output.values()
            if status == "predicted"]


class Audit:
    def __init__(self, root):
        self.root = root
        rows = read_csv(root / "results/simulator/method_comparison.csv")
        self.reader = unique_rows([r for r in rows if r["result_kind"] == "prediction"],
                                  ("context_id", "method_id"))
        self.checked = set()

    def csv(self, relative):
        return read_csv(self.root / relative)

    def compare(self, context, method, pairs, assigned, predicted, quality=None):
        key = (context, method)
        if key in self.checked:
            raise ValueError(f"method checked twice: {key}")
        reader = self.reader[key]
        computed = metrics(pairs)
        computed.update(expected_assigned_n=assigned, predicted_n=predicted,
                        coverage=len(pairs) / assigned)
        if quality is not None:
            passed, failed, unassessed, pass_pairs = quality
            computed.update(quality_assessed_n=passed + failed, quality_pass_n=passed,
                            quality_fail_n=failed, quality_unassessed_n=unassessed)
            if reader["quality_pass_mae_seconds"]:
                for name, value in metrics(pass_pairs).items():
                    if name in {"mae_seconds", "medae_seconds", "log1p_mae_seconds", "r2_seconds"}:
                        computed["quality_pass_" + name] = value
        for name, value in computed.items():
            if value is None:
                if reader.get(name, "") != "":
                    raise ValueError(f"published value for undefined metric: {key} {name}")
                continue
            if reader.get(name, "") == "":
                if name in {"expected_assigned_n", "predicted_n", "metric_n", "coverage", "mae_seconds", "r2_seconds"} or name.startswith("quality_"):
                    raise ValueError(f"missing published field: {key} {name}")
                # Older component-model summaries did not publish every metric.
                # Recompute them, but do not invent a comparison value.
                continue
            if name.endswith("_n"):
                if int(reader[name]) != value:
                    raise ValueError(f"count mismatch: {key} {name}")
            else:
                equal(value, reader[name], f"{key} {name}")
        self.checked.add(key)

    def dense(self):
        targets = self.csv("data/simulator/statevector/targets/dense_hash_targets.csv")
        target_map = unique_rows(targets, ("precision", "qasm_sha256"))
        for precision in ("fp32", "fp64"):
            methods = defaultdict(list)
            for path in sorted((self.root / "data/simulator/statevector/predictions" / precision).glob("*/fold_*/predictions.csv")):
                for row in read_csv(path):
                    target = target_map[(precision, row["qasm_sha256"])]
                    if row["fold"] != target["outer_fold"] or target["assigned_core_oof"] != "True":
                        raise ValueError("dense target/fold/core mismatch")
                    equal(number(row["target_seconds"]), number(target["target_seconds"]), "dense target")
                    equal(statistics.median(json.loads(target["session_medians_seconds"])),
                          number(target["target_seconds"]), "dense session reduction")
                    if row.get("seed_prediction_min_seconds", ""):
                        low, high = number(row["seed_prediction_min_seconds"]), number(row["seed_prediction_max_seconds"])
                        if not low <= number(row["prediction_seconds"]) <= high:
                            raise ValueError("dense prediction outside recorded seed bounds")
                        equal(high - low, row["seed_prediction_range_seconds"], "dense seed range")
                    methods[row["method_id"]].append(row)
            for method, rows in methods.items():
                unique_rows(rows, ("qasm_sha256",))
                pairs = [(number(r["target_seconds"]), number(r["prediction_seconds"]))
                         for r in rows if r["status"] == "ok"]
                self.compare("cudaq_dense_statevector_" + precision, method, pairs, len(rows), len(pairs))

    def fixed_mps(self):
        targets = unique_rows(self.csv("data/simulator/mps/recovered_targets/fixed_chi16_targets.csv"),
                              ("source_qasm_sha256",))
        for relative, methods in (
            ("fixed_recovered/aggregate", ("train_fold_median", "ridge_alpha_1", "graph_median_three_seeds")),
            ("residual_recovered", ("family_residual_median_three_seeds", "family_agnostic_median_three_seeds")),
        ):
            rows = self.csv(f"data/simulator/mps/{relative}/five_fold_oof_predictions.csv")
            if set(unique_rows(rows, ("source_qasm_sha256",))) != set(targets):
                raise ValueError("fixed MPS prediction/target identities differ")
            for row in rows:
                target = targets[(row["source_qasm_sha256"],)]
                if row["fold"] != target["fold"] or row["quality_status_separate_audit_only"] != target["quality_status_separate"]:
                    raise ValueError("fixed MPS fold/quality mismatch")
                equal(row["target_seconds"], target["target_seconds"], "fixed MPS target")
                for method in methods:
                    if method.endswith("median_three_seeds"):
                        prefix = method.removesuffix("median_three_seeds")
                        seed_median(row, f"pred_{method}_seconds", [f"pred_{prefix}seed_{s}_seconds" for s in SEEDS])
            for method in methods:
                pairs = [(number(r["target_seconds"]), number(r[f"pred_{method}_seconds"])) for r in rows]
                pass_pairs = [pair for pair, r in zip(pairs, rows) if r["quality_status_separate_audit_only"] == "quality_pass"]
                self.compare("cudaq_mps_fp64_bond16_warm_state_execution_seconds", method,
                             pairs, len(rows), len(pairs), (len(pass_pairs), len(rows) - len(pass_pairs), 0, pass_pairs))

    def aer(self):
        base = "data/simulator/aer/azizov/"
        labels = self.csv(base + "labels_core.csv")
        label_map = unique_rows(labels, ("panel_member_id",))
        by_hash = defaultdict(list)
        for row in labels:
            by_hash[row["source_sha256"]].append(row)
        seed_rows = self.csv(base + "oof_predictions_by_seed.csv")
        seeds = unique_rows(seed_rows, ("panel_member_id", "view", "seed"))
        methods = defaultdict(list)
        for name in ("classical_oof_predictions.csv", "oof_predictions_median.csv"):
            for row in self.csv(base + name):
                label = label_map[(row["panel_member_id"],)]
                if row["source_sha256"] != label["source_sha256"] or row["fold"] != label["fold"]:
                    raise ValueError("Aer label/hash/fold mismatch")
                aliases = by_hash[row["source_sha256"]]
                if len({r["fold"] for r in aliases}) != 1:
                    raise ValueError("Aer exact hash crosses folds")
                equal(row["target_seconds"], statistics.median(number(r["observed_seconds"]) for r in aliases), "Aer hash-median target")
                equal(row["member_observed_seconds"], label["observed_seconds"], "Aer member target")
                if name == "oof_predictions_median.csv":
                    values = [seeds[(row["panel_member_id"], row["view"], str(s))] for s in SEEDS]
                    if any(r["status"] != "predicted" or r["fold"] != row["fold"] for r in values):
                        raise ValueError("Aer incomplete/mismatched seed set")
                    equal(statistics.median(number(r["prediction_seconds"]) for r in values),
                          row["prediction_seconds"], "Aer seed median")
                methods[row["method_id"]].append(row)
        for method, rows in methods.items():
            if {r["panel_member_id"] for r in rows} != {r["panel_member_id"] for r in labels}:
                raise ValueError("Aer method omits assigned members")
            pairs = collapse_aliases(rows)
            self.compare("qiskit_aer_fake_sherbrooke_noisy_opt1", method, pairs, len(by_hash), len(pairs))

    def joint_mps(self):
        base = "data/simulator/mps/joint/"
        attempts = read_jsonl(self.root / (base + "measurement/attempt_records.jsonl"))
        unique_rows(attempts, ("source_qasm_sha256", "max_bond", "session_id"))
        cells = defaultdict(list)
        for row in attempts:
            cells[(row["source_qasm_sha256"], str(row["max_bond"]))].append(row)
        for name, suffix in (("oof_predictions.csv", "c44"),
                             ("family_holdout_diagnostic_oof_predictions.csv", "family_holdout")):
            methods = defaultdict(list)
            for row in self.csv(base + "analysis/" + name):
                sessions = cells[(row["source_qasm_sha256"], row["max_bond"])]
                if len(sessions) != 3:
                    raise ValueError("joint MPS missing measurement session")
                observed = all(r["status"] == "ok" for r in sessions)
                if observed != (row["target_status"] == "runtime_observed"):
                    raise ValueError("joint MPS target availability differs from measurement")
                if observed:
                    actual = statistics.median(statistics.median(number(v) for v in r["warm_seconds"]) for r in sessions)
                    equal(actual, row["target_runtime_seconds"], "joint MPS measured target")
                    passed = all(number(r["fidelity"]) >= 0.99 for r in sessions)
                    if passed != (row["target_quality_pass"] == "True"):
                        raise ValueError("joint MPS measured quality mismatch")
                seed_median(row, "runtime_median_seed_seconds", [f"runtime_seed_{s}_seconds" for s in SEEDS])
                equal(statistics.mean(number(row[f"quality_seed_{s}_probability"]) for s in SEEDS),
                      row["quality_mean_seed_probability"], "joint MPS quality seed mean")
                methods[row["method_id"]].append(row)
            for method, rows in methods.items():
                if set(unique_rows(rows, ("source_qasm_sha256", "max_bond"))) != set(cells):
                    raise ValueError("joint MPS assigned identities differ")
                observed = [r for r in rows if r["target_status"] == "runtime_observed"]
                pairs = [(number(r["target_runtime_seconds"]), number(r["runtime_median_seed_seconds"])) for r in observed]
                passed = sum(r["target_quality_pass"] == "True" for r in observed)
                reader_method = method + "_joint_rung_" + suffix
                self.compare("cudaq_mps_fp64_chi_2_4_8_16_32_64", reader_method, pairs,
                             len(rows), len(rows), (passed, len(observed) - passed, len(rows) - len(observed), []))

    def maestro(self):
        base = "data/simulator/qcsim/maestro/"
        targets = unique_rows(self.csv(base + "measurement/panel_targets.csv"), ("source_qasm_sha256",))
        rows = self.csv(base + "evaluation/oof_predictions.csv")
        if set(unique_rows(rows, ("source_qasm_sha256",))) != set(targets):
            raise ValueError("QCSim assigned identities differ")
        methods = {
            "maestro_style_cpu_sv_component_optimizer_off_adaptation": "component_prediction_status",
            "outer_train_median": "outer_train_median_status",
            "nested_grouped_ridge": "nested_grouped_ridge_status",
            "source_dag_graph_adaptation_cuda": "source_dag_graph_status",
        }
        for row in rows:
            target = targets[(row["source_qasm_sha256"],)]
            if row["fold"] != target["outer_fold"] or row["target_status"] != target["target_status"]:
                raise ValueError("QCSim target status/fold mismatch")
            if row["target_status"] == "runtime_observed":
                equal(row["runtime_seconds"], target["runtime_seconds"], "QCSim target")
                seed_median(row, "pred_source_dag_graph_adaptation_cuda_seconds", [f"pred_graph_seed_{s}_seconds" for s in SEEDS])
        for method, status in methods.items():
            predicted = [r for r in rows if r[status] == "available"]
            pairs = [(number(r["runtime_seconds"]), number(r[f"pred_{method}_seconds"]))
                     for r in predicted if r["target_status"] == "runtime_observed"]
            self.compare("qcsim_statevector_cpu_fp64_1000_shots", method, pairs, len(rows), len(predicted))

    def companions(self):
        rows = read_jsonl(self.root / "data/simulator/analog/raw_records.jsonl")
        unique_rows(rows, ("row_id", "session_id", "stage", "repetition"))
        cells = defaultdict(list)
        for row in rows:
            cells[row["row_id"]].append(row)
        pass_cells = 0
        for records in cells.values():
            if sorted(r["stage"] for r in records) != ["first_execute", "network_build", "warm_execute", "warm_execute"]:
                raise ValueError("analog incomplete cell")
            passes = [number(r["quality"]["candidate_fidelity"]) >= 0.99 for r in records]
            if len(set(passes)) != 1:
                raise ValueError("analog inconsistent cell quality")
            pass_cells += passes[0]
            for row in records:
                number(row["elapsed_seconds"])
                if number(row["quality"]["reference_fidelity_256_vs_512"]) < 0.9999:
                    raise ValueError("analog reference quality failed")
                if (row["status"] == "ok") != passes[0]:
                    raise ValueError("analog quality/status mismatch")
        if (len(rows), len(cells), pass_cells) != (72, 18, 17):
            raise ValueError("analog record/cell/quality counts differ")
        ratios = self.csv("data/simulator/tensor_network/analysis/paired_ratio_rows.csv")
        unique_rows(ratios, ("panel_member_id", "session_id"))
        if len(ratios) != 603:
            raise ValueError("TN paired-row count differs")
        for row in ratios:
            ratio = number(row["runtime_est_s"]) / number(row["warm_contract_median_s"])
            equal(ratio, row["ratio"], "TN estimate/warm ratio")
            equal(math.log(ratio), row["log_ratio"], "TN log ratio")
            equal(abs(math.log(ratio)), row["abs_log_ratio"], "TN absolute log ratio")
        return len(cells), len(ratios)

    def run(self):
        self.dense()
        self.fixed_mps()
        self.aer()
        self.joint_mps()
        self.maestro()
        if self.checked != set(self.reader):
            raise ValueError(f"unchecked predictor rows: {sorted(set(self.reader) - self.checked)}")
        analog, tn = self.companions()
        return len(self.checked), analog, tn


if __name__ == "__main__":
    predictors, analog, tn = Audit(ROOT).run()
    print(f"Matched saved-label OOF metrics for {predictors} predictor rows.")
    print(f"Checked quality for {analog} analog cells and ratios for {tn} TN pairs.")
    print("No fitting, inference or new measurements. QPU aggregate scores not recomputed.")
