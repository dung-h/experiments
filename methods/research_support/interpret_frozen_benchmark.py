#!/usr/bin/env python3
"""Saved-evidence interpretation only. No fitting, inference or engine execution.

Prints apply_patch input for new derived tables; never writes source artifacts.
The original-support route re-extracts deterministic static circuit features
from pinned public QASM. This is preprocessing verification, not simulation.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/benchmark_interpretation/v1"
PARENT = ROOT / "artifacts/real_qpu/qualified_logical_panel"
GEN = ROOT / "artifacts/real_qpu/generalization"
GRID = GEN / "complete_grid_v1/results/full_grid_20261011"
DEEP = ROOT / "artifacts/benchmark_deep_dive/tables"
UPSTREAM = ROOT.parent / "Quantum-Execution-Time-Prediction"
REV = "32c392a6ece276f1ff046d4e30052d0571ff6dc6"
PINS = {}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    data = path.read_bytes()
    PINS[str(path)] = sha(data)
    return data


def js(path):
    return json.loads(read(path))


def rows(path):
    return list(csv.DictReader(io.StringIO(read(path).decode())))


def blob(relative):
    return subprocess.check_output(["git", "-C", str(UPSTREAM), "show", f"{REV}:{relative}"])


def csv_text(values):
    assert values
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(values[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(values)
    return text.getvalue()


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def patch(files):
    print("*** Begin Patch")
    for name, text in files.items():
        path = OUT / name
        if path.exists():
            if path.read_text() == text:
                continue
            raise RuntimeError("refuse_derived_overwrite:" + str(path))
        print("*** Add File: " + str(path))
        for line in text.splitlines():
            print("+" + line)
    print("*** End Patch")


def original_support():
    """Reconstruct support, without loading the 12-GiB derived PyG pickle."""
    import mali_full_features as features
    from qiskit import QuantumCircuit

    extractor, code_pins = features.load_pinned_global_extractor()
    names = list(extractor["create_feature_dict"](
        QuantumCircuit.from_qasm_str('OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];')))
    assert len(names) == 51
    timing = {}
    for device in ("washington", "sherbrooke", "osaka", "kyoto"):
        relative = f"data/{device}_time_taken.csv"
        data = blob(relative)
        assert read(UPSTREAM / relative) == data
        timing[device] = list(csv.DictReader(io.StringIO(data.decode())))
        for row in timing[device]:
            row["circuit_name"] = Path(row.get("circuit_name", row.get("quantum_circuit", ""))).stem
    sim_names = [r["circuit_name"] for r in timing["washington"]]
    assert sim_names == [r["circuit_name"] for r in timing["sherbrooke"]]
    assert len(sim_names) == 1510 and len(set(sim_names)) == 1510
    circuit_names = sorted(set(sim_names) | {r["circuit_name"] for d in ("osaka", "kyoto") for r in timing[d]})
    vectors = {}
    raw_source_digests = {}
    for i, name in enumerate(circuit_names):
        relative = f"data/quantum_circuits/{name}.qasm"
        data = blob(relative)
        assert read(UPSTREAM / relative) == data
        raw_source_digests[relative] = sha(data)
        circuit = QuantumCircuit.from_qasm_str(data.decode())
        result = extractor["create_feature_dict"](circuit)
        assert list(result) == names
        vectors[name] = np.asarray(list(result.values()), dtype=float)
        assert np.isfinite(vectors[name]).all()
        if i % 200 == 0:
            print(f"static feature audit {i}/{len(circuit_names)}", file=sys.stderr, flush=True)
    sim = np.array([vectors[name] for name in sim_names] * 2)
    real_names = [r["circuit_name"] for d in ("osaka", "kyoto") for r in timing[d]]
    real = np.array([vectors[name] for name in real_names])
    assert sim.shape == (3020, 51) and real.shape == (340, 51)
    support = []
    for j, name in enumerate(names):
        support.append(dict(raw_index=j, feature_name=name,
            original_sim_nonzero_rows=int(np.count_nonzero(sim[:, j])),
            original_real_nonzero_rows=int(np.count_nonzero(real[:, j])),
            original_sim_sum=float(sim[:, j].sum()), original_real_sum=float(real[:, j].sum()),
            retained_original_sim_refinement=bool(sim[:, j].sum() > 0),
            retained_original_real_refinement=bool(real[:, j].sum() > 0)))
    result = dict(upstream_revision=REV, feature_names=names, support=support,
        simulator_rows=3020, simulator_unique_named_circuits=1510, real_rows=340,
        sim_effective_width=int(np.count_nonzero(sim.sum(axis=0) > 0)),
        real_effective_width=int(np.count_nonzero(real.sum(axis=0) > 0)),
        source_qasm_sha256=raw_source_digests, source_code_pins=code_pins,
        scope="Recomputed pinned static features; author processed tensors were not published.",
        source_pins=PINS)
    patch({"original_corpus_support.json": json_text(result)})


def upstream_pipeline():
    paths = ["main.ipynb", "data_preparation/alr.py", "data_preparation/execution.py",
             "model/circs.py", "model/builder.py", "model/transformer_model.py",
             "model/train.py", "model/cross_validation.py", "model/parameter/default/config.yaml"]
    code = {p: blob(p) for p in paths}
    notebook = json.loads(code["main.ipynb"])
    cells = notebook["cells"]
    def record(i):
        cell = cells[i]
        return dict(zero_based_cell=i, cell_type=cell["cell_type"], source="".join(cell["source"]),
                    stdout="".join("".join(o.get("text", [])) for o in cell.get("outputs", [])))
    sim_record, real_record = record(16), record(48)
    sim_indices = ast.literal_eval(sim_record["stdout"].strip())
    real_indices = ast.literal_eval(real_record["stdout"].strip())
    assert len(sim_indices) == 41 and len(real_indices) == 12
    original = js(OUT / "original_corpus_support.json")
    assert sim_indices == [r["raw_index"] for r in original["support"] if r["retained_original_sim_refinement"]]
    model = ast.parse(code["model/transformer_model.py"])
    klass = next(n for n in model.body if isinstance(n, ast.ClassDef) and n.name == "Simple_Model")
    init = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert ast.literal_eval(init.args.defaults[-1]) == 41
    return {"upstream_pipeline_evidence.json": json_text(dict(upstream_revision=REV,
        git_blob_sha256={p: sha(data) for p,data in code.items()},
        simulator_refinement_notebook=sim_record, real_refinement_notebook=real_record,
        simulator_recorded_retained_indices=sim_indices, real_recorded_retained_indices=real_indices,
        real_recomputed_published_340_indices=[r["raw_index"] for r in original["support"] if r["retained_original_real_refinement"]],
        model_default_global_width=41, active_learning_padding="alr padding function pads node rows only, not globals",
        notebook_relevant_cells=[record(i) for i in [17,20,24,25,26,29,46,49,52,60,62,64,65,67,68]],
        interpretation="Simulator stored mask agrees; real notebook stored12 differs from published340 support40. Author executed real tensor/schema not established; do not silently promote either as historical truth.",
        no_notebook_cell_executed=True, no_checkpoint_loaded=True))}


def retained(values):
    return np.flatnonzero(values.sum(axis=0) > 0).tolist()


def transform_row(label, stage, method, seed, fit, test, tfm, x, index, names, path):
    values = x[[index[i] for i in fit]]
    t = tfm["global"]
    keep = retained(values)
    assert keep == t["retained_indices"], (label, stage, keep, t["retained_indices"])
    z = values[:, keep]
    assert np.allclose(z.mean(axis=0), t["mean"], rtol=1e-10, atol=1e-10)
    std = z.std(axis=0, ddof=1)
    assert np.allclose(std, t["std"], rtol=1e-9, atol=1e-10)
    assert (std > 1e-6).tolist() == t["variable_mask"]
    test_values = x[[index[i] for i in test]]
    dropped = sorted(set(range(51)) - set(keep))
    held_only = [j for j in dropped if np.count_nonzero(test_values[:, j])]
    return dict(case_id=label, stage=stage, method_id=method, seed=seed,
        fit_rows=len(fit), test_rows=len(test), raw_width=51, effective_tensor_width=len(keep),
        variable_width=int(sum(t["variable_mask"])),
        retained_indices_json=json.dumps(keep), dropped_names_json=json.dumps([names[j] for j in dropped]),
        held_out_only_names_json=json.dumps([names[j] for j in held_only]),
        held_out_only_nonzero_rows_json=json.dumps({names[j]: int(np.count_nonzero(test_values[:, j])) for j in held_only}),
        transform_path=str(path.relative_to(ROOT)), transform_sha256=PINS[str(path)])


def feature_tables():
    original = js(OUT / "original_corpus_support.json")
    names = original["feature_names"]
    panel = rows(PARENT / "panel.csv")
    assert len(panel) == 4515
    ids = [r["canonical_observation_id"] for r in panel]
    index = {i: j for j, i in enumerate(ids)}
    x = np.array([[float(r[f"g{j:02d}"]) for j in range(51)] for r in panel])
    outer = rows(PARENT / "outer_splits.csv")
    inner = rows(PARENT / "inner_splits.csv")
    transforms = []
    for fold in range(5):
        test = sorted(r["canonical_observation_id"] for r in outer if int(r["outer_fold"]) == fold)
        fit = sorted(r["canonical_observation_id"] for r in inner if int(r["outer_fold"]) == fold and int(r["inner_fold"]) != 0)
        p = PARENT / f"neural/fold_{fold}/transform.json"
        tfm = js(p)
        transforms.append(transform_row(f"outer_fold:{fold}", "gradient_fit_inner_1_2_3", "graph_and_mlp", "42;1234;31415", fit, test, tfm, x, index, names, p))
    split = js(GEN / "splits.json")
    source = js(GEN / "complete_grid_v1/attempts/desktop_identity_tolerant_recovery_20261011e/preparation.json")
    for case in split["cases"]:
        if not case["default_execute"]:
            continue
        c = case["case_id"]
        neural = Path(source["sources_by_case"][c]["neural_root"])
        inner_fold = case["neural_selection_inner_fold"]
        selection = next(t for t in case["inner"] if t["inner_fold"] == inner_fold)
        for stage, fit_ix in [("selection", selection["fit"]), ("final_refit", case["train"])]:
            fit = sorted(split["row_index"][j]["canonical_observation_id"] for j in fit_ix)
            test = sorted(split["row_index"][j]["canonical_observation_id"] for j in case["test"])
            for method in ("mali_logical_graph", "mali_global_mlp"):
                for seed in (42, 1234, 31415):
                    p = neural / f"{stage}/{method}/seed_{seed}/transform.json"
                    tfm = js(p)["transform"]
                    transforms.append(transform_row(c, stage, method, seed, fit, test, tfm, x, index, names, p))
    assert len(transforms) == 257
    ledger = []
    for j, name in enumerate(names):
        source_n = {s: int(np.count_nonzero(x[[k for k, r in enumerate(panel) if r["source_id"] == s], j])) for s in sorted({r["source_id"] for r in panel})}
        keep_in = [k for k, t in enumerate(transforms) if j in json.loads(t["retained_indices_json"])]
        hold_in = sorted({t["case_id"] + "/" + t["stage"] for t in transforms if name in json.loads(t["held_out_only_names_json"])})
        ledger.append({**original["support"][j], "feature_class": "gate_count" if j < 44 else "width_depth" if j < 46 else "SupermarQ_structural",
            "unified_nonzero_rows": int(np.count_nonzero(x[:, j])),
            "mali_nonzero_rows": source_n["mali_real_qpu"], "qonductor_nonzero_rows": source_n["qonductor_single_circuit_ibm"], "qpack_nonzero_rows": source_n["qpack_mcp"],
            "outer_retained_folds_json": json.dumps([t["case_id"] for t in transforms[:5] if j in json.loads(t["retained_indices_json"])]),
            "retained_transform_row_indices_json": json.dumps(keep_in),
            "held_out_only_case_stages_json": json.dumps(hold_in),
            "can_reach_any_current_model": bool(keep_in)})
    concentrations = []
    meta = {r["canonical_observation_id"]: r for r in split["row_index"]}
    suspect = [j for j in range(51) if not original["support"][j]["retained_original_sim_refinement"] or not original["support"][j]["retained_original_real_refinement"]]
    for j in suspect:
        for s in sorted({r["source_id"] for r in panel}):
            for dimension in ("family", "backend"):
                counters = Counter(meta[r["canonical_observation_id"]][dimension] for k, r in enumerate(panel) if r["source_id"] == s and x[k, j] != 0)
                concentrations.append(dict(raw_index=j, feature_name=names[j], source_id=s, dimension=dimension,
                    nonzero_rows=sum(counters.values()), counts_json=json.dumps(dict(counters), sort_keys=True),
                    interpretation="support only; not predictive or causal importance"))
    widths = Counter((t["stage"], t["effective_tensor_width"]) for t in transforms)
    summary = dict(status="PASS", checked_transform_files=len(transforms), raw_width=51,
        parent_widths=[t["effective_tensor_width"] for t in transforms[:5]],
        transform_width_counts=[dict(stage=s, width=w, count=n) for (s, w), n in sorted(widths.items())],
        unified_zero_names=[names[j] for j in range(51) if not np.count_nonzero(x[:, j])],
        decision="B_REPORTING_FIX_ONLY", no_rerun_for_feature_count=True,
        raw_schema_support_rule_matches_upstream=True, source_pins=PINS)
    return {"global_feature_ledger.csv": csv_text(ledger), "effective_input_widths.csv": csv_text(transforms),
        "masked_feature_concentration.csv": csv_text(concentrations), "feature_validation.json": json_text(summary)}


def metric(values):
    y = np.asarray([float(v["actual_seconds"]) for v in values])
    p = np.asarray([float(v["predicted_seconds"]) for v in values])
    assert len(y) and np.isfinite(y).all() and np.isfinite(p).all()
    e = p - y
    den = np.sum((y - y.mean()) ** 2)
    return dict(n=len(y), mae_seconds=float(np.abs(e).mean()), medae_seconds=float(np.median(np.abs(e))),
        rmse_seconds=float(np.sqrt(np.mean(e ** 2))), r2_seconds=float(1 - np.sum(e ** 2) / den) if den > 0 else "",
        signed_error_mean_seconds=float(e.mean()), overprediction_rows=int(np.count_nonzero(e > 0)),
        p99_absolute_error_seconds=float(np.quantile(np.abs(e), .99)), max_absolute_error_seconds=float(np.abs(e).max()))


def skills_for(scope, predictions, references, extra=None):
    groups = defaultdict(list)
    for r in predictions:
        if r.get("status", "predicted") == "predicted" and r.get("predicted_seconds", "") != "":
            groups[r["method_id"]].append(r)
    refgroups = {k: {v["canonical_observation_id"]: v for v in values} for k, values in references.items()}
    result = []
    for method, values in sorted(groups.items()):
        for ref, byid in refgroups.items():
            shared = [v for v in values if v["canonical_observation_id"] in byid]
            if not shared:
                continue
            expected = [byid[v["canonical_observation_id"]] for v in shared]
            assert len({v["canonical_observation_id"] for v in shared}) == len(shared)
            assert all(abs(float(a["actual_seconds"]) - float(b["actual_seconds"])) < 1e-10 for a, b in zip(shared, expected))
            a, b = metric(shared), metric(expected)
            result.append(dict(scope=scope, method_id=method, reference_id=ref, **a,
                reference_mae_seconds=b["mae_seconds"], skill=1 - a["mae_seconds"] / b["mae_seconds"] if b["mae_seconds"] else "",
                reference_context_advantage=ref in {"outer_train_source_median", "train_source_median", "eligible_source_median"},
                **(extra or {})))
    return result


def skill_tables():
    oof = rows(PARENT / "summary/oof_predictions.csv")
    refs = rows(DEEP / "qpu_baseline_predictions_v1.csv")
    refgroups = defaultdict(list)
    for r in refs:
        if r["method_id"] != "outer_train_mean":
            refgroups[r["method_id"]].append(r)
    # Selector is inner-validation-selected, not a winner chosen on outer-test.
    selector = [r for r in oof if r["method_id"] == "qonductor_family_selector"]
    if not selector:
        selector = [r for r in oof if "selector" in r["method_id"]]
    if selector:
        refgroups["inner_selected_classical_regression"] = selector
    skill = skills_for("qpu_pooled", oof, refgroups)
    sources = sorted({r["source_id"] for r in oof})
    for s in sources:
        skill.extend(skills_for("qpu_source:" + s, [r for r in oof if r["source_id"] == s], refgroups))
    # Macro skill = 1 - macro MAE_method / macro MAE_reference. Do not average R2 or skill ratios.
    macro = defaultdict(list)
    for r in skill:
        if r["scope"].startswith("qpu_source:"):
            macro[r["method_id"], r["reference_id"]].append(r)
    macros = []
    for (m, ref), vs in sorted(macro.items()):
        if len(vs) == 3 and len({v["scope"] for v in vs}) == 3:
            a = float(np.mean([v["mae_seconds"] for v in vs])); b = float(np.mean([v["reference_mae_seconds"] for v in vs]))
            macros.append(dict(method_id=m, reference_id=ref, sources=3, method_macro_mae_seconds=a,
                reference_macro_mae_seconds=b, source_balanced_skill=1-a/b,
                interpretation="equal-source macro; not iid observations or context parity"))
    split = js(GEN / "splits.json")
    ledger = {r["canonical_observation_id"]: r for r in rows(PARENT / "panel.csv")}
    source = js(GEN / "complete_grid_v1/attempts/desktop_identity_tolerant_recovery_20261011e/preparation.json")
    seedrows = rows(GRID / "neural_seed_attempts.csv")
    gen = []
    for case in split["cases"]:
        if not case["default_execute"]:
            continue
        cid = case["case_id"]
        ids = {split["row_index"][i]["canonical_observation_id"] for i in case["test"]}
        train = [ledger[split["row_index"][i]["canonical_observation_id"]] for i in case["train"]]
        gm = float(np.median([float(t["target_seconds"]) for t in train]))
        sm = {s: float(np.median([float(t["target_seconds"]) for t in train if t["source_id"] == s])) for s in {t["source_id"] for t in train}}
        rr = defaultdict(list)
        for i in sorted(ids):
            row = ledger[i]
            for name, value in (("train_global_median", gm), ("train_source_median", sm.get(row["source_id"], gm))):
                rr[name].append(dict(canonical_observation_id=i, actual_seconds=row["target_seconds"], predicted_seconds=value))
        cpu = rows(Path(source["sources_by_case"][cid]["cpu_root"]) / "attempts.csv")
        cpu = [r for r in cpu if r["canonical_observation_id"] in ids]
        selector = [r for r in cpu if "selector" in r["method_id"]]
        if selector:
            rr["inner_selected_classical_regression"] = selector
        ag = defaultdict(list)
        for r in seedrows:
            if r["case_id"] == cid and r["status"] == "predicted":
                ag[r["method_id"], r["canonical_observation_id"]].append(r)
        neural = []
        for (m, i), vs in ag.items():
            assert len(vs) == 3 and {int(v["seed"]) for v in vs} == {42,1234,31415}
            neural.append({**vs[0], "method_id": m, "predicted_seconds": float(np.median([float(v["predicted_seconds"]) for v in vs]))})
        gen.extend(skills_for(cid, cpu + neural, rr, dict(experiment=case["experiment"], estimator="saved_cpu_or_neural_three_seed_median")))
    sim = rows(DEEP / "simulator_prediction_view_v1.csv")
    simskill = []
    for context in sorted({r["context_id"] for r in sim}):
        part = [r for r in sim if r["context_id"] == context]
        # Simulator hash IDs are the common-row key, not real-QPU observation IDs.
        for r in part:
            r["canonical_observation_id"] = r.get("canonical_observation_id") or r.get("hash_id", r.get("group_id", ""))
        if any(not r["canonical_observation_id"] for r in part):
            raise RuntimeError("simulator_identity_key_missing")
        rr = defaultdict(list)
        for r in part:
            if r["method_id"] in {"outer_train_median", "train_fold_median", "nested_grouped_ridge", "ridge_alpha_1", "classical_xgboost_transpiled", "random_forest"}:
                rr[r["method_id"]].append(r)
        simskill.extend(skills_for(context, part, rr))
    analytical = []
    for path in sorted(DEEP.glob("analytical_*_predictions_v1.csv")):
        values = rows(path)
        for r in values:
            r["method_id"] = r["variant"]
        rr = {ref: [r for r in values if r["variant"] == ref]
              for ref in ("eligible_intercept", "eligible_median", "eligible_source_median")}
        analytical.extend(skills_for(path.name.removeprefix("analytical_").removesuffix("_predictions_v1.csv"), values, rr))
    return {"qpu_predictive_skill.csv": csv_text(skill), "qpu_source_balanced_skill.csv": csv_text(macros),
        "generalization_predictive_skill.csv": csv_text(gen), "simulator_predictive_skill.csv": csv_text(simskill),
        "analytical_predictive_skill.csv": csv_text(analytical),
        "skill_validation.json": json_text(dict(status="PASS", no_models_fitted=True, no_inference=True,
            parent_methods=sorted({r["method_id"] for r in oof}), cases=21,
            baseline_sources="Existing parent references; RQ4 train-only medians replayed from frozen partitions.",
            rows={"qpu":len(skill),"macro":len(macros),"generalization":len(gen),"simulator":len(simskill)},source_pins=PINS))}


def generalization_tables():
    split = js(GEN / "splits.json")
    panel = rows(PARENT / "panel.csv")
    ledger = {r["canonical_observation_id"]: r for r in panel}
    mask_rows = rows(GRID / "reader_tables/feature_masking.csv")
    masks = {(r["case_id"], r["method_id"]): r for r in mask_rows}
    parent = defaultdict(dict)
    for r in rows(PARENT / "summary/oof_predictions.csv"):
        if r["status"] == "predicted":
            parent[r["method_id"]][r["canonical_observation_id"]] = r
    seeds = rows(GRID / "neural_seed_attempts.csv")
    values = defaultdict(list)
    for r in seeds:
        values[r["case_id"], r["method_id"], r["canonical_observation_id"]].append(r)
    output, fields = [], []
    for c in split["cases"]:
        if not c["default_execute"]:
            continue
        train = [ledger[split["row_index"][i]["canonical_observation_id"]] for i in c["train"]]
        test = [ledger[split["row_index"][i]["canonical_observation_id"]] for i in c["test"]]
        x = np.asarray([[float(r[f"g{j:02d}"]) for j in range(51)] for r in train])
        z = np.asarray([[float(r[f"g{j:02d}"]) for j in range(51)] for r in test])
        std = x.std(axis=0, ddof=1); varying = std > 1e-6
        standardized = np.zeros_like(z)
        standardized[:, varying] = (z[:, varying]-x[:, varying].mean(axis=0))/std[varying]
        max_z = np.abs(standardized).max(axis=1)
        outside = (z < x.min(axis=0)) | (z > x.max(axis=0))
        unseen = (x.sum(axis=0) == 0) & (z != 0).any(axis=0)
        for j in range(51):
            fields.append(dict(case_id=c["case_id"], raw_index=j, train_nonzero_rows=int(np.count_nonzero(x[:,j])),
                test_nonzero_rows=int(np.count_nonzero(z[:,j])), train_min=float(x[:,j].min()), train_max=float(x[:,j].max()),
                test_min=float(z[:,j].min()), test_max=float(z[:,j].max()),
                outside_train_range_test_rows=int(outside[:,j].sum()),
                max_absolute_train_standardized_test_value=float(np.abs(standardized[:,j]).max()) if varying[j] else "",
                unseen_test_only=bool(unseen[j]), diagnostic="descriptive support; no fitted attribution"))
        for method in ("mali_logical_graph", "mali_global_mlp"):
            ps = []
            for r in test:
                identity = r["canonical_observation_id"]
                vs = values[c["case_id"], method, identity]
                assert len(vs) == 3
                ps.append(dict(canonical_observation_id=identity, actual_seconds=r["target_seconds"],
                    predicted_seconds=float(np.median([float(v["predicted_seconds"]) for v in vs]))))
            m = metric(ps)
            parent_values = [parent[method][r["canonical_observation_id"]] for r in test]
            p = metric(parent_values)
            mask = masks[c["case_id"], method]
            same = int(mask["test_rows_with_exact_fit_collision"])
            output.append(dict(case_id=c["case_id"], experiment=c["experiment"], method_id=method,
                **m, structural_groups=len({r["group_id"] for r in test}),
                known_group_test_rows=len(c["known_structural_group_test"]), new_group_test_rows=len(c["new_structural_group_test"]),
                exact_consumed_input_collision_rows=same,
                effective_final_global_width=51-len(json.loads(mask["global_dropped_raw_indices_json"])),
                held_out_only_global_indices_json=json.dumps(np.flatnonzero(unseen).tolist()),
                median_max_absolute_train_z=float(np.median(max_z)), max_absolute_train_z=float(max_z.max()),
                outside_any_raw_train_range_rows=int(outside.any(axis=1).sum()),
                parent_oof_same_rows_mae_seconds=p["mae_seconds"],
                holdout_to_parent_mae_ratio=m["mae_seconds"]/p["mae_seconds"],
                degradation_interpretation="descriptive: changed train population AND parent/refit training procedure",
                test_sources_json=json.dumps(dict(Counter(r["source_id"] for r in test)),sort_keys=True),
                test_shots_json=json.dumps(dict(Counter(float(r["shots"]) for r in test)),sort_keys=True)))
    return {"generalization_decomposition.csv": csv_text(output), "generalization_field_support.csv": csv_text(fields)}


def shots():
    panel = rows(PARENT / "panel.csv")
    split = js(GEN / "splits.json")
    meta = {r["canonical_observation_id"]: r for r in split["row_index"]}
    result = []
    for dims in [("source_id",), ("source_id", "backend"), ("source_id", "family")]:
        buckets = defaultdict(list)
        for r in panel:
            rr = {**r, **meta[r["canonical_observation_id"]]}
            buckets[tuple(rr[k] for k in dims)].append(rr)
        for key, values in sorted(buckets.items()):
            counts = Counter(float(r["shots"]) for r in values)
            result.append(dict(stratum="/".join(dims), key="/".join(key), observation_rows=len(values),
                structural_groups=len({r["group_id"] for r in values}), distinct_shot_values=len(counts),
                min_shots=min(counts), max_shots=max(counts), shot_counts_json=json.dumps(dict(sorted(counts.items()))),
                systematic_qpu_shot_holdout_status="NOT ANSWERED"))
    return {"shot_diversity.csv": csv_text(result)}


def interpretation_tables():
    """Explicit scientific judgments, not inferred from test-performance ranking."""
    fidelity = rows(DEEP / "method_fidelity_v1.csv")[:6]
    for row in fidelity:
        if row["method"] == "Scholten nominal":
            row["method"] = "Scholten-inspired generalized proxy"
            row["original_formulation"] = "Original: M*S*D_eff/C; D_eff=E[depth(kernel)]/E[depth(QV(v,v))]*v; v=ceil(sqrt(2*D*n))"
            row["available_inputs"] = "Current generic compiled wire depth and nominal throughput; original kernel/template/QV-ensemble semantics not verified on any qualified row"
            row["fidelity"] = "proxy; original verified eligibility=0/4515; not evaluable"
        row["native_direct_service_target_alignment"] = (
            "learned observed service label; heterogeneous source boundaries remain" if row["method"] in {"Ma–Li QPU", "Qonductor-style regression"}
            else "no: native timing/resource quantity differs; deterministic shot scaling and label-fitted bridges are separately named")
        row["normalization_rule"] = (
            "train-fit feature normalization; not a timing-to-service calibration" if row["method"] in {"Ma–Li QPU", "Qonductor-style regression"}
            else "raw; deterministic shot multiplication where declared; affine/log-affine fits on eligible outer-train only; not original-estimator fidelity")

    dims = ["Estimator fidelity", "Target validity/alignment", "In-distribution accuracy",
            "Predictive value beyond simple baselines", "Feature/representation contribution", "Generalization",
            "Scaling/extrapolation", "Reliability/uncertainty/applicability/coverage"]
    evidence = {
        "parent": "artifacts/real_qpu/qualified_logical_panel/summary/method_comparison.csv",
        "skill": "artifacts/benchmark_interpretation/v1/qpu_predictive_skill.csv",
        "macro": "artifacts/benchmark_interpretation/v1/qpu_source_balanced_skill.csv",
        "feature": "artifacts/benchmark_interpretation/v1/feature_validation.json",
        "contribution": "artifacts/benchmark_interpretation/v1/feature_contribution.md",
        "rq4": "artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011/method_metrics.csv",
        "paired": "artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011/reader_tables/paired_comparisons.csv",
        "shots": "artifacts/benchmark_interpretation/v1/shot_diversity.csv",
        "analytical": "artifacts/benchmark_deep_dive/tables/analytical_controls_v1.csv",
        "fidelity": "artifacts/benchmark_interpretation/v1/track_a_fidelity.csv",
        "sim": "artifacts/benchmark_deep_dive/tables/simulator_metrics_v1.csv",
        "simpair": "artifacts/benchmark_deep_dive/tables/simulator_paired_v1.csv",
        "simskill": "artifacts/benchmark_interpretation/v1/simulator_predictive_skill.csv",
        "slices": "artifacts/benchmark_deep_dive/tables/simulator_slices_v1.csv",
        "quality": "artifacts/benchmark_deep_dive/tables/mps_selection_safety_v1.csv",
        "fixedquality": "artifacts/benchmark_deep_dive/tables/mps_quality_metrics_v1.csv",
        "tn": "artifacts/benchmark_deep_dive/tables/tn_same_plan_slices_v1.csv",
        "analog": "artifacts/benchmark_deep_dive/tables/pasqal_stage_quality_v1.csv",
        "width": "experiments/simulator_runtime_v2/README.md",
        "launch": "artifacts/cudaq_runtime/launch_scaling_v1_20260922/analysis/REPORT.md",
    }
    # Eight ordered judgments per scope. ANSWERED always means within the stated context.
    scopes = [
      ("A", "Ma–Li Graph/global MLP", [
       ("ANSWERED","Architecture adaptation; raw51/support masks verified; nominal T1/T2 and training changes disclosed","fidelity;feature"),
       ("PARTIAL","Observed source service/execution labels, not a harmonized pulse or provider clock; reconstructed inputs qualified","fidelity;parent"),
       ("ANSWERED","Grouped OOF accuracy on the fixed 4515; not a random IID population guarantee","parent"),
       ("ANSWERED","Positive pooled skill vs global median; small/unresolved advantage vs classical selector and ranking reversals under reweighting","skill;macro"),
       ("PARTIAL","Usable historical reliance and retrained controls; no matched full4515 topology-only/global-block ablation","contribution;feature"),
       ("PARTIAL","All21 declared cases scored; failures and collisions visible; shots/hardware-causal transfer not established","rq4;paired;shots"),
       ("PARTIAL","Two QPack size probes with3/2groups only; no general QPU shot/width extrapolation law","rq4;shots"),
       ("PARTIAL","Coverage and group-bootstrap error uncertainty exist; not calibrated prospective prediction intervals or production guarantees","parent;paired")]),
      ("A", "Qonductor six regressors / budget selector", [
       ("ANSWERED","Pinned native counters and six families; bounded307-candidate search differs from full original search","fidelity"),
       ("PARTIAL","Trains observed service labels; compiled/snapshot and mitigation limitations disclosed","fidelity;parent"),
       ("ANSWERED","Same outer-test rows; six families and inner-only selector retained","parent"),
       ("ANSWERED","Matched median controls and source-balanced comparisons; no test-selected winner","skill;macro"),
       ("PARTIAL","Five native counters available; no local retrained individual-counter attribution on4515","fidelity;contribution"),
       ("PARTIAL","21 CPU cases; strong QNN/Random references, but source/context shifts confounded","rq4;shots"),
       ("PARTIAL","Two size probes; source shot diversity is not a shot-held-out evaluation","rq4;shots"),
       ("PARTIAL","Tail/coverage and conditional bootstrap; finite search and source dependence constrain reliability","parent;paired")]),
      ("A", "QCRE estimator component", [
       ("ANSWERED","Pinned original estimate_runtime invoked on admitted unitary views; not full QCRE compiler workflow","fidelity"),
       ("ANSWERED","Native unitary critical-path seconds do not equal observed service; shot scaling omits reset/readout/host terms","fidelity"),
       ("ANSWERED","Raw and separate calibrated errors on shared qualified rows retained","parent;analytical"),
       ("ANSWERED","Affine gains must be compared with eligible intercept/source medians, not raw alone","analytical;skill"),
       ("PARTIAL","Historical critical/static summaries; no new controlled nominal-layout-duration decomposition","contribution"),
       ("PARTIAL","21 CPU-case outputs exist; nominal snapshots are not isolated hardware shift","rq4;fidelity"),
       ("PARTIAL","Size probes available; multiplying shots is an assumption, not validated shot extrapolation","rq4;shots"),
       ("PARTIAL","Coverage/units checked; historical routing/calibration and omitted operations remain uncertain","fidelity;analytical")]),
      ("A", "Qiskit scheduled-duration reference", [
       ("ANSWERED","API timing reference on frozen nominal Target/circuit","fidelity"),
       ("ANSWERED","Scheduled one-shot, shot-scaled and calibrated service outputs are different clocks","fidelity"),
       ("ANSWERED","Native and bridged errors recorded separately","parent;analytical"),
       ("ANSWERED","Train-only eligible controls expose intercept/context contributions","analytical;skill"),
       ("PARTIAL","Timing signal assessed; detailed duration-term contributions not isolated on4515","contribution;analytical"),
       ("PARTIAL","21 CPU cases; reference not a learned hardware adaptation model","rq4;fidelity"),
       ("PARTIAL","Size cases; no observed shot scaling validation","rq4;shots"),
       ("PARTIAL","Deterministic timing applicability known; service uncertainty not modeled by API","fidelity;analytical")]),
      ("A", "Scholten original", [
       ("ANSWERED","Original kernel/QV effective-depth definition identified; zero verified eligible rows","fidelity"),
       ("PARTIAL","Original quantum-kernel timing boundary not verified against generic service labels","fidelity"),
       ("NOT ANSWERED","No faithful original-method accuracy on this generic cohort","fidelity;parent"),
       ("NOT ANSWERED","Proxy results cannot answer original estimator incremental skill","fidelity;analytical"),
       ("NOT ANSWERED","No qualifying kernel/QV ensemble to attribute original D_eff","fidelity"),
       ("NOT ANSWERED","Proxy CPU holdouts are not original Scholten transfer evidence","fidelity;rq4"),
       ("NOT ANSWERED","No original-method size/shot extrapolation test","fidelity;shots"),
       ("PARTIAL","Original not-evaluable reason explicit; empirical uncertainty absent","fidelity")]),
      ("A", "Scholten-inspired generalized proxy", [
       ("ANSWERED","Compiled wire depth plus nominal throughput; explicitly not original D_eff","fidelity"),
       ("ANSWERED","Nominal throughput output and label-fitted service bridge kept separate","fidelity"),
       ("ANSWERED","3399 covered rows;1116 missing throughput retained in denominator","parent;analytical"),
       ("ANSWERED","Affine/log-affine improve on eligible global median but not source median; context advantage disclosed","analytical"),
       ("PARTIAL","Depth/throughput association not isolated from source/calibration","analytical;contribution"),
       ("PARTIAL","CPU holdouts only on eligible contexts; no original or isolated hardware claim","rq4;fidelity"),
       ("PARTIAL","Some size-probe evidence; no systematic shot holdout","rq4;shots"),
       ("PARTIAL","Coverage explicit; nominal throughput definition/date and service boundary limit deployment","fidelity;analytical")]),
      ("A", "Hyb-HANAS identifiable component", [
       ("ANSWERED","Reliability-adjusted identifiable cost component; not entire NAS training estimator","fidelity"),
       ("ANSWERED","Survival-adjusted useful-work/resource cost is not literal observed execution time","fidelity"),
       ("ANSWERED","Raw finite4175 and log-cost Ridge4367 with distinct outputs/coverage","parent;analytical"),
       ("ANSWERED","Eligible median/source controls outperform several calibrated cost variants","analytical;skill"),
       ("PARTIAL","Cost/survival/log association measured; no controlled per-term ablation","fidelity;analytical"),
       ("PARTIAL","CPU-case predictions where finite; no full-Hyb hardware/architecture generalization","rq4;fidelity"),
       ("PARTIAL","Local size probes; no validated useful-shot scaling law","rq4;shots"),
       ("PARTIAL","Zero-survival/overflow explicit; finite cost is not guaranteed service predictability","fidelity;analytical")]),
      ("B", "CUDA-Q dense FP32/FP64", [
       ("ANSWERED","Context-specific Ma–Li-style controls and internal tabular/work baselines, not paper score reproduction","sim;fidelity"),
       ("ANSWERED","Warm sample wall-clock per precision; no equation to MPS/state/scalar clocks","sim"),
       ("ANSWERED","150 shared exact hashes; RF lower MAE than graph/MLP in each precision","sim;simpair"),
       ("ANSWERED","RF positive skill vs train median; graph worse than matched metadata MLP","simskill;simpair"),
       ("PARTIAL","Metadata/control comparisons, not topology-only retrained ablation","sim;simpair"),
       ("PARTIAL","Grouped folds/slices; no broad device/family/shot OOD grid","slices;launch"),
       ("PARTIAL","Historical width and controlled launch studies distinct from this common panel","width;launch"),
       ("PARTIAL","Conditional hash-bootstrap and coverage; one local GPU/software stack","simpair;sim")]),
      ("B", "CUDA-Q fixed chi16 MPS", [
       ("ANSWERED","Local graph/global/family adaptations; fixed FP64 bond/state operation","sim;fidelity"),
       ("ANSWERED","Warm get_state, not sampling; quality assessed separately","sim;fixedquality"),
       ("ANSWERED","150 hashes graphMAE0.2961s vs Ridge1.2757s","sim"),
       ("ANSWERED","Graph positive skill vs median/Ridge on matched hashes","simskill;simpair"),
       ("PARTIAL","Family-conditioned vs agnostic contrast exists; graph topology attribution absent","simpair;contribution"),
       ("PARTIAL","Grouped/sliced fixed-chi evidence, not arbitrary unseen entanglement success","slices;fixedquality"),
       ("PARTIAL","Ladder companion explores chi but fixed-context runtime is not a scaling law","quality;sim"),
       ("PARTIAL","147/150 quality pass; runtime prediction on a completed state does not guarantee required fidelity","fixedquality;simpair")]),
      ("B", "Azizov noisy Aer source/hybrid/transpiled", [
       ("ANSWERED","Paper-described model adaptation; authors full encoding unavailable; one local backend/opt context","fidelity"),
       ("ANSWERED","Post-transpile warm Aer execution, FakeSherbrooke Opt1 shots1024","sim;fidelity"),
       ("ANSWERED","150 hashes with exact shared rows and3 GNN seeds","sim"),
       ("ANSWERED","Transpiled XGBoost0.2970s matches GNN0.3008s; paired interval includes zero","simskill;simpair"),
       ("ANSWERED","Source vs transpiled local representation contrast; not individual-gate causality","simpair"),
       ("PARTIAL","Grouped hashes/family slices; original two backend/four-level transfer not reproduced","slices;fidelity"),
       ("NOT ANSWERED","No common-panel Aer width/shot extrapolation test","sim;slices"),
       ("PARTIAL","Bootstrap and attempted coverage; no prospective error guarantees or other-machine claims","simpair;sim")]),
      ("B", "Family-aware joint MPS ladder", [
       ("ANSWERED","Joint family/quality/runtime adaptation to local chi ladder, not original Quantum Rings threshold","fidelity"),
       ("ANSWERED","Warm get_state runtime and fidelity probability are separate outputs","quality;sim"),
       ("ANSWERED","840 shared rungs/140 complete hashes scored","sim"),
       ("ANSWERED","Conditioning has unresolved/worse point MAE vs matched agnostic ablation","simpair"),
       ("ANSWERED","Matched family-conditioning ablation; no resolved local runtime advantage","simpair"),
       ("PARTIAL","Family-held-out comparison exists, but does not establish safe unseen-family selection","simpair;quality"),
       ("PARTIAL","Six chi rungs tested; width/shot/hardware extrapolation not established","quality;slices"),
       ("ANSWERED","Unsafe selector finding retained:64/140 quality failures and10 unassessed, not operationally safe","quality")]),
      ("B", "Maestro-style QCSim CPU component", [
       ("ANSWERED","Operation/sampling calibration adaptation; optimizer off; no full closed Composer selector","fidelity"),
       ("ANSWERED","Engine-reported execution across fresh processes; not persistent warm API wall-clock","fidelity;sim"),
       ("ANSWERED","143 supported hashes, componentMAE0.0001037s","sim"),
       ("ANSWERED","Positive matched skill vs median/Ridge; no automatic multi-engine selection claim","simskill;simpair"),
       ("PARTIAL","Component structure justified, individual calibration-term attribution unresolved","fidelity;sim"),
       ("PARTIAL","Grouped hash interpolation only; unseen engine/hardware transfer absent","slices;fidelity"),
       ("NOT ANSWERED","No saved width/shot/hardware holdout validating this resolved component","sim;slices"),
       ("PARTIAL","Supported set/interval known; original engine selection and persistent warm clock untested","sim;simpair")]),
      ("B", "cuTensorNet same-plan native RUNTIME_EST", [
       ("ANSWERED","Native API output on pinned selected plan; internal optimizer not reverse-engineered","fidelity;tn"),
       ("ANSWERED","All-slice scalar contraction estimate vs same-plan warm contraction, not whole simulator API","tn;fidelity"),
       ("ANSWERED","603/612 plan rows;201 members/188 hashes; ratio slices retained","tn"),
       ("NOT ANSWERED","No matched train-only simple predictor on this native plan population","tn;sim"),
       ("PARTIAL","Slicing/workspace/size diagnostics descriptive, internal cost-term causality unknown","tn"),
       ("PARTIAL","Within-local-plan applicability; no hardware/operation universal transfer","tn;fidelity"),
       ("PARTIAL","Plan-size/slicing regimes examined; not all-network/hardware extrapolation","tn"),
       ("PARTIAL","Estimate/warmmedian3.734;coverage explicit, opaque internals and first/build/search costs excluded","tn;fidelity")]),
      ("B", "Pasqal analog companion", [
       ("ANSWERED","Documentation-based analog MPS/TDVP companion, not digital method leaderboard","fidelity;analog"),
       ("ANSWERED","Host MPSBackend.run excluding quality capture; analog pulse/context distinct","analog;fidelity"),
       ("PARTIAL","18 cells measured; no promoted predictive coefficient model","analog"),
       ("NOT ANSWERED","No matched cross-validated simple reference/identified analytical predictor","analog;fidelity"),
       ("PARTIAL","Bond/quality observations only; N²chi³/N³chi² coefficients not identifiable","analog;fidelity"),
       ("NOT ANSWERED","No new-analog-family/hardware held-out estimator evaluation","analog"),
       ("PARTIAL","Limited grid, unidentifiable work coefficients; no reliable scaling extrapolation","analog;fidelity"),
       ("PARTIAL","17/18 quality gates pass;chi16 quench fails; timing success alone insufficient","analog")]),
      ("B", "Historical PyTorch dense width study", [
       ("ANSWERED","Internally implemented kernel/baselines, distinct engine from CUDA-Q common panel","width"),
       ("ANSWERED","Historical PyTorch kernel wall-clock only","width"),
       ("ANSWERED","208 rows grouped HGB logR2.9794 in its recorded context","width"),
       ("ANSWERED","Work-scaled analytical Ridge beats HGB on saved width holdout","width"),
       ("PARTIAL","Complexity structure contrast, no causal physical-launch isolation","width;launch"),
       ("PARTIAL","Grouped/width splits only for that kernel","width"),
       ("ANSWERED","Recorded width holdout logR2.8754 Ridge vs.7295HGB; not universal law","width"),
       ("PARTIAL","OOM/tails recorded; context-specific precision/workload limits","width;launch")]),
    ]
    matrix = []
    for track, scope, judgments in scopes:
        assert len(judgments) == len(dims)
        for dim, (status, conclusion, keys) in zip(dims, judgments):
            paths = [evidence[k] for k in keys.split(";")]
            matrix.append(dict(track=track, method_or_context=scope, dimension=dim,
                status=status, conclusion=conclusion, evidence_paths=";".join(paths),
                scope_rule="Conditional local answer; ANSWERED is not universal validity"))
    return {"track_a_fidelity.csv": csv_text(fidelity), "benchmark_question_matrix.csv": csv_text(matrix)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["original-support", "upstream-pipeline", "features", "skill", "shots", "generalization", "questions"])
    p.add_argument("--file", help="Render one named output only to keep patch transfer bounded.")
    args = p.parse_args()
    if args.stage == "original-support":
        original_support()
    else:
        outputs = {"features": feature_tables, "skill": skill_tables, "shots": shots,
                   "generalization": generalization_tables, "questions": interpretation_tables,
                   "upstream-pipeline": upstream_pipeline}[args.stage]()
        patch({args.file: outputs[args.file]} if args.file else outputs)


if __name__ == "__main__":
    main()
