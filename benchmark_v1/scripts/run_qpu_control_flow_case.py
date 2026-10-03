#!/usr/bin/env python3
"""Preflight or fit the frozen fold-0 V4 single-case adaptation.

Preflight uses only CPU file validation. Fit requires CUDA and fresh weights,
does not read a test target until all seed predictions have been produced,
and never changes the historical V3/V4 contracts or results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROTOCOL = ROOT / "benchmark_v1/protocol/qpu_control_flow_case.json"
KINDS = ("operation", "control", "true_branch", "false_branch", "merge")
GLOBAL = ("active_width", "structural_depth", "one_qubit_count", "two_qubit_count",
          "swap_like_count", "measurement_count", "shots")
CASE_QASM_SHA = "abc2b5725d3e34544b56b5c70fc444a38d6cee85003a0f3359400ddf49899ecf"
PROTOTYPE_DIGEST = "94dc83063aad6447ca2940cdb3c36a9cabab2ea1d68142e38339e8b0b9523bf9"
INPUT_ROLES = {
    "canonical": "artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv",
    "outer": "artifacts/benchmark_v2/real_qpu/unified_outer_split_v2.csv",
    "features": "artifacts/benchmark_v3/recovery_real_qpu_20260929_v3/c134_feature_sidecar_v3.csv",
    "panel": "artifacts/benchmark_v3/s71_graph_panel_v3_large_20260930/graph_row_manifest.csv",
    "prototype": "artifacts/benchmark_v3/real_qpu/dynamic_control_representation_probe_v1r2/prototype.json",
    "encoder": "benchmark_v1/scripts/s71_graph_v4_control_flow.py",
    "report": "artifacts/benchmark_v3/real_qpu/dynamic_control_representation_probe_v1r2/report.json",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path, key: str) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate observation IDs: {path}")
    return result


def validate_partition(canonical: dict, outer: dict, features: dict, panel: dict,
                       case_id: str, expected_rows: int = 8767) -> tuple[list[str], list[str]]:
    ids = set(canonical)
    if len(ids) != expected_rows or any(set(x) != ids for x in (outer, features, panel)):
        raise ValueError("canonical/split/feature/graph identity mismatch")
    if case_id not in ids or int(outer[case_id]["outer_fold"]) != 0:
        raise ValueError("case is not assigned to outer fold 0")
    train = sorted(cid for cid in ids if int(outer[cid]["outer_fold"]) != 0)
    test = sorted(ids - set(train))
    tg = {outer[cid]["unified_leakage_group_id"] for cid in train}
    vg = {outer[cid]["unified_leakage_group_id"] for cid in test}
    if tg & vg:
        raise ValueError("frozen outer split leaks groups")
    for cid in train:
        if panel[cid]["status"] != "eligible":
            raise ValueError(f"training graph unavailable: {cid}")
        gates = json.loads(features[cid].get("gate_counts_json") or "{}")
        if any(int(gates.get(k, 0)) for k in ("if_else", "while_loop", "for_loop", "switch_case")):
            raise ValueError("dynamic training support changed; review the case protocol")
        y = float(canonical[cid]["target_seconds"])
        if not math.isfinite(y) or y < 0:
            raise ValueError(f"invalid training target: {cid}")
    return train, test


def load_inputs(input_root: Path) -> dict:
    protocol = json.loads(PROTOCOL.read_text())
    for relative, expected in protocol["input_pins"].items():
        if sha(input_root / relative) != expected:
            raise ValueError(f"input hash mismatch: {relative}")
    paths = {role: input_root / relative for role, relative in INPUT_ROLES.items()}
    canonical = read_rows(paths["canonical"], "canonical_row_id")
    outer = read_rows(paths["outer"], "canonical_observation_id")
    features = read_rows(paths["features"], "canonical_row_id")
    panel = read_rows(paths["panel"], "canonical_observation_id")
    prototype = json.loads(paths["prototype"].read_text())
    case_id = protocol["evaluation"]["case_id"]
    train, test = validate_partition(canonical, outer, features, panel, case_id)
    report = json.loads(paths["report"].read_text())
    if canonical[case_id]["qasm_bytes_sha256"] != CASE_QASM_SHA or report["original_qasm_digest"] != CASE_QASM_SHA:
        raise ValueError("prototype/canonical case QASM identity mismatch")
    validate_prototype(prototype)
    spec = importlib.util.spec_from_file_location("pinned_s79_encoder", paths["encoder"])
    signed_encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(signed_encoder)
    vocab = {name: i for i, name in enumerate(sorted({name for name, kind in zip(prototype["names"], prototype["kinds"]) if kind == "operation"}))}
    vocab["__UNK_GATE__"] = len(vocab)
    validate_layout_parity(prototype, vocab, signed_encoder)
    graph_dir = paths["panel"].parent / "graphs"
    graph_paths = {panel[cid]["representation_digest"]: graph_dir / panel[cid]["graph_file"]
                   for cid in train}
    if any(not path.is_file() for path in graph_paths.values()):
        raise FileNotFoundError("training graph pickles missing; restore the recorded graph panel")
    return dict(protocol=protocol, canonical=canonical, outer=outer, features=features,
                panel=panel, prototype=prototype, train=train, test=test,
                graph_paths=graph_paths, case_id=case_id)


def validate_prototype(prototype: dict) -> None:
    if (prototype["format"] != "dynamic-control-structural-prototype-v1r2"
            or prototype["format_version"] != "v1r2" or prototype["num_nodes"] != 409
            or prototype["control_flow_operation_count"] != 10):
        raise ValueError("unexpected dynamic prototype format/support")
    if prototype["unsupported_constructs"] or prototype["runtime_target_read"]:
        raise ValueError("prototype is unsupported or contains target-derived encoding")
    if prototype["authorized_for_learned_model"] is not False:
        raise ValueError("historical prototype authorization was altered")
    keys = ("format", "format_version", "num_qubits", "num_clbits", "names", "kinds",
            "arity", "parameter_presence", "q0", "q1", "q2", "q3", "edge_src", "edge_dst", "edge_kind", "branch_records")
    fields = {key: prototype[key] for key in keys}
    fields["condition_structures"] = [item["condition_structure"] for item in prototype["branch_records"]]
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != PROTOTYPE_DIGEST or prototype["prototype_digest"] != digest:
        raise ValueError("prototype semantic digest mismatch")
    edges = set(zip(prototype["edge_src"], prototype["edge_dst"], prototype["edge_kind"]))
    if len(prototype["branch_records"]) != 10:
        raise ValueError("unexpected branch record count")
    for item in prototype["branch_records"]:
        control, true, false, merge = (int(item[k]) for k in ("control_node", "true_branch_node", "false_branch_node", "merge_node"))
        if (control, true, "control_to_true_branch") not in edges or (control, false, "control_to_false_branch") not in edges:
            raise ValueError("missing explicit control edges")
        if item["merge_policy"] != "per_qubit_terminal" or not item["false_branch_explicitly_emitted"]:
            raise ValueError("branch merge/false-branch policy mismatch")
        if item["false_branch_operation_count"] != 0 or item["selected_branch"] is not None or item["branch_probability"] is not None:
            raise ValueError("case branch semantics mismatch")
        true_terms, false_terms = item["true_terminals_by_qubit"], item["false_terminals_by_qubit"]
        if not true_terms or {x["qubit"] for x in true_terms} != {x["qubit"] for x in false_terms}:
            raise ValueError("per-wire terminal sets differ")
        if any(int(x["node"]) != false for x in false_terms):
            raise ValueError("empty false branch does not merge from false node")
        for terms, kind in ((true_terms, "true_branch_to_merge"), (false_terms, "false_branch_to_merge")):
            if any((int(x["node"]), merge, kind) not in edges for x in terms):
                raise ValueError("missing per-wire merge edge")


def validate_layout_parity(record: dict, vocab: dict, signed_encoder) -> None:
    encoded = signed_encoder.encode_v4_node_layout(record, vocab)
    actual = node_rows(record, vocab)
    if len(actual) != len(encoded["rows"]):
        raise ValueError("node count differs from signed S79 encoder")
    for index, (row, expected) in enumerate(zip(actual, encoded["rows"])):
        prefix = expected["gate_one_hot"] + expected["arity_one_hot"] + [expected["parameter_presence"]]
        suffix = expected["kind_one_hot"] + expected["condition_channels"]
        # S79's helper returns the new channels, but omits the four inherited
        # positions. Independently check their signed min/mean/max/present rule.
        wires = [int(record[key][index]) for key in ("q0", "q1", "q2", "q3")
                 if int(record[key][index]) >= 0]
        denominator = max(1, int(record["num_qubits"]) - 1)
        positions = ([min(wires) / denominator, sum(wires) / len(wires) / denominator,
                      max(wires) / denominator, 1.0] if wires else [0.0] * 4)
        if (row[:len(prefix)] != prefix or row[-9:] != suffix
                or any(not math.isclose(a, b, abs_tol=1e-12)
                       for a, b in zip(row[len(prefix):len(prefix) + 4], positions))):
            raise ValueError("node layout differs from signed S79 encoder")


def condition_values(structure: dict, num_clbits: int) -> list[float]:
    value = structure.get("comparison_value")
    compare = float(value) if isinstance(value, (int, float)) else 0.0
    index = structure.get("classical_bit_index")
    size = structure.get("register_size") or 0
    return [float(bool(structure.get("predicate_present"))), compare,
            0.0 if index is None else float(index) / max(1, num_clbits), math.log1p(size)]


def node_rows(record: dict, vocab: dict) -> list[list[float]]:
    """S79 feature order; structural names never expand the gate vocabulary."""
    for forbidden in ("target_seconds", "y", "source_id", "backend", "branch_probability", "selected_branch"):
        if forbidden in record:
            raise ValueError(f"forbidden graph field: {forbidden}")
    names = record["names"]
    kinds = record.get("kinds", ["operation"] * len(names))
    if len(kinds) != len(names) or any(k not in KINDS for k in kinds):
        raise ValueError("invalid node kinds")
    conditions = {int(r["control_node"]): r["condition_structure"]
                  for r in record.get("branch_records", [])}
    rows = []
    scale = max(1, int(record["num_qubits"]) - 1)
    params = record.get("parameter_presence", record.get("parameter"))
    for i, name in enumerate(names):
        row = [0.0] * (len(vocab) + 18)
        row[vocab.get(name, vocab["__UNK_GATE__"])] = 1.0
        offset = len(vocab)
        row[offset + min(int(record["arity"][i]), 3)] = 1.0
        row[offset + 4] = float(params[i])
        pos = [int(record[k][i]) / scale for k in ("q0", "q1", "q2", "q3")
               if int(record[k][i]) >= 0]
        if pos:
            row[offset + 5:offset + 9] = [min(pos), sum(pos) / len(pos), max(pos), 1.0]
        row[offset + 9 + KINDS.index(kinds[i])] = 1.0
        if kinds[i] == "control":
            row[offset + 14:offset + 18] = condition_values(conditions[i], int(record["num_clbits"]))
        rows.append(row)
    return rows


def fit(data: dict, output: Path, canary_only: bool = False) -> dict:
    import numpy as np
    import torch
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import TransformerConv, global_mean_pool
    from benchmark_v1.scripts.run_mps_fixed_chi16_runtime_adaptation_v1 import exclusive_compute_lock

    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    cfg = data["protocol"]["training"]

    class Model(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.conv = torch.nn.ModuleList([TransformerConv(width, 64),
                                            TransformerConv(64, 64), TransformerConv(64, 64)])
            self.gf = torch.nn.Sequential(torch.nn.Linear(7, 64), torch.nn.ReLU(),
                                          torch.nn.Linear(64, 64), torch.nn.ReLU())
            self.head = torch.nn.Sequential(torch.nn.Linear(128, 512), torch.nn.ReLU(),
                                            torch.nn.Linear(512, 512), torch.nn.ReLU(),
                                            torch.nn.Linear(512, 128), torch.nn.ReLU(),
                                            torch.nn.Linear(128, 1))

        def forward(self, batch):
            x = batch.x
            for conv in self.conv:
                x = torch.relu(conv(x, batch.edge_index))
            return self.head(torch.cat([global_mean_pool(x, batch.batch),
                                       self.gf(batch.global_features)], dim=1)).reshape(-1)

    with exclusive_compute_lock():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU fallback is forbidden")
        torch.set_num_threads(cfg["cpu_threads"])
        records = {}
        graph_hashes = {}
        for digest, path in data["graph_paths"].items():
            graph_hashes[digest] = sha(path)
            with path.open("rb") as f:
                record = pickle.load(f)
            # This loader accepts only the recorded trusted local pickle panel.
            if record.get("kinds") and set(record["kinds"]) != {"operation"}:
                raise ValueError("unexpected structural nodes in static training panel")
            records[digest] = record
        names = sorted({name for rec in records.values() for name in rec["names"]})
        vocab = {name: i for i, name in enumerate(names)}
        vocab["__UNK_GATE__"] = len(vocab)
        globals_ = {cid: np.log1p(np.asarray([max(0.0, float(data["features"][cid][k]))
                                            for k in GLOBAL], dtype=np.float32))
                    for cid in data["train"] + [data["case_id"]]}
        train_values = np.stack([globals_[cid] for cid in data["train"]])
        mean, std = train_values.mean(0), train_values.std(0)
        std[std < 1e-6] = 1.0
        topology = {}
        for digest, record in records.items():
            topology[digest] = (torch.tensor(node_rows(record, vocab), dtype=torch.float32),
                                torch.tensor([record["edge_src"], record["edge_dst"]], dtype=torch.long))
        case = data["prototype"]
        case_x = torch.tensor(node_rows(case, vocab), dtype=torch.float32)
        case_edges = torch.tensor([case["edge_src"], case["edge_dst"]], dtype=torch.long)

        def observation(cid):
            x, edges = topology[data["panel"][cid]["representation_digest"]]
            return Data(x=x, edge_index=edges,
                        global_features=torch.tensor((globals_[cid] - mean) / std).reshape(1, -1),
                        y=torch.tensor(math.log1p(float(data["canonical"][cid]["target_seconds"]))))

        dataset = {cid: observation(cid) for cid in data["train"]}
        case_data = Data(x=case_x, edge_index=case_edges,
                         global_features=torch.tensor((globals_[data["case_id"]] - mean) / std).reshape(1, -1))
        output.mkdir(parents=True)
        identity = dict(protocol_sha256=sha(PROTOCOL), runner_sha256=sha(Path(__file__)),
                        input_pins=data["protocol"]["input_pins"], graph_file_sha256=graph_hashes,
                        vocab=vocab, global_mean=mean.tolist(), global_std=std.tolist(),
                        cuda_device=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                        cuda_version=torch.version.cuda, train_rows=len(dataset), outer_fold=0,
                        dynamic_train_rows=0, five_fold_score=False, status="running",
                        automatic_resume_supported=False,
                        python_version=sys.version, numpy_version=np.__version__)
        import importlib.metadata
        identity["torch_geometric_version"] = importlib.metadata.version("torch-geometric")
        (output / "run_manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
        # Panel preflight precedes this gate. Ten train-only forward/backward
        # passes check the real tensor layout; no optimizer update or test label.
        torch.manual_seed(cfg["seeds"][0]); torch.cuda.manual_seed_all(cfg["seeds"][0])
        smoke = Model(len(vocab) + 18).cuda()
        for cid in data["train"][:10]:
            smoke.zero_grad()
            batch = Batch.from_data_list([dataset[cid]]).cuda()
            prediction = smoke(batch)
            loss = torch.nn.functional.mse_loss(prediction, batch.y.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("CUDA canary loss is nonfinite")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in smoke.parameters()):
                raise RuntimeError("CUDA canary gradient is nonfinite")
        with torch.no_grad():
            if not torch.isfinite(smoke(Batch.from_data_list([case_data]).cuda())).all():
                raise RuntimeError("target-free case forward is nonfinite")
        del smoke
        canary = dict(status="PASS", train_graphs=10, optimizer_steps=0, case_target_read=False,
                      protocol_sha256=identity["protocol_sha256"], cuda_device=identity["cuda_device"])
        (output / "canary.json").write_text(json.dumps(canary, indent=2) + "\n")
        if canary_only:
            identity["status"] = "canary_only_no_fit"
            (output / "run_manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
            return canary
        predictions = []
        for seed in cfg["seeds"]:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = Model(len(vocab) + 18).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
            rng = random.Random(seed)
            technical_batches = 0
            for epoch in range(cfg["epochs"]):
                model.train()
                order = list(data["train"]); rng.shuffle(order)
                for start in range(0, len(order), cfg["effective_batch"]):
                    ids = order[start:start + cfg["effective_batch"]]
                    optimizer.zero_grad()
                    micro = []
                    node_count = 0

                    def backward(chunk):
                        nonlocal technical_batches
                        batch = Batch.from_data_list([dataset[cid] for cid in chunk]).cuda()
                        pred = model(batch)
                        loss = torch.nn.functional.mse_loss(pred, batch.y.reshape(-1), reduction="sum") / len(ids)
                        if not torch.isfinite(loss):
                            raise RuntimeError("nonfinite training loss")
                        loss.backward(); technical_batches += 1

                    for cid in ids:
                        size = dataset[cid].num_nodes
                        if micro and node_count + size > cfg["node_microbatch_budget"]:
                            backward(micro); micro = []; node_count = 0
                        micro.append(cid); node_count += size
                    if micro:
                        backward(micro)
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise RuntimeError("nonfinite training gradient")
                    optimizer.step()
                torch.save({"seed": seed, "completed_epoch": epoch + 1, "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "protocol_sha256": identity["protocol_sha256"],
                            "python_rng": random.getstate(), "order_rng": rng.getstate(),
                            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all()},
                           output / f"checkpoint_seed_{seed}.pt")
            if technical_batches < 10:
                raise RuntimeError("technical checkpoint did not cover ten microbatches")
            model.eval()
            with torch.no_grad():
                pred = float(torch.expm1(model(Batch.from_data_list([case_data]).cuda())).clamp_min(0).cpu().item())
            if not math.isfinite(pred):
                raise RuntimeError("nonfinite case prediction")
            predictions.append(dict(seed=seed, prediction_seconds=pred))
        # Persist predictions before using the held-out case label for any metric.
        (output / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
        observed = float(data["canonical"][data["case_id"]]["target_seconds"])
        values = [x["prediction_seconds"] for x in predictions]
        median = float(np.median(values))
        result = dict(case_id=data["case_id"], observed_seconds=observed,
                      median_prediction_seconds=median, absolute_error_seconds=abs(median - observed),
                      seed_predictions=[{**x, "absolute_error_seconds": abs(x["prediction_seconds"] - observed)} for x in predictions],
                      seed_prediction_range_seconds=max(values) - min(values), dynamic_train_rows=0,
                      claim=data["protocol"]["claim"], status="completed_single_case_only")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        identity["status"] = result["status"]
        identity["output_sha256"] = {path.name: sha(path) for path in output.iterdir()
                                     if path.is_file() and path.name != "run_manifest.json"}
        (output / "run_manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["preflight", "canary", "fit"], default="preflight")
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.action != "preflight" and args.output_dir is not None and args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    data = load_inputs(args.input_root.resolve())
    if args.action == "preflight":
        print(json.dumps(dict(status="PASS", train_rows=len(data["train"]),
                              withheld_fold0_rows=len(data["test"]), scored_cases=1,
                              train_dynamic_rows=0, input_pins_valid=True,
                              signed_layout_parity=True, prototype_invariants_valid=True,
                              training_performed=False, gpu_used=False), indent=2))
    else:
        if args.output_dir is None:
            parser.error("fit requires a new --output-dir")
        print(json.dumps(fit(data, args.output_dir.resolve(), args.action == "canary"), indent=2))


if __name__ == "__main__":
    main()
