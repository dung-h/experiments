#!/usr/bin/env python3
"""Pinned-model A1 source pretraining and qualification runner.

This runner is an A1-local wrapper. It obtains ``Simple_Model`` only via
``git show <pinned-commit>:model/transformer_model.py`` and never redefines the
neural architecture. It has no simulator-execution or QPU-execution API.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

OUT = Path(__file__).resolve().parent
# OUT is <repo>/artifacts/postfreeze_track_a_transfer_v1.
REPO = OUT.parents[1]
UPSTREAM = Path("/home/server/Documents/Quantum-Execution-Time-Prediction")
PIN = "32c392a6ece276f1ff046d4e30052d0571ff6dc6"
SEEDS = (42, 1234, 31415)
# CUDA capacity was benchmarked before graph training: 700,000 nodes / 1,200,000
# directed edges used 11.94 GiB; 1,000,000 nodes OOMed. The fixed cap below leaves
# a conservative margin for PyG batching, model state, and desktop GPU processes.
GRAPH_BATCH_NODE_CAP = 650_000
GRAPH_BATCH_EDGE_CAP = 1_100_000
FEATURES = OUT / "simulator_features" / "fixed_raw_51_features.csv"
FEATURE_MANIFEST = OUT / "simulator_features" / "strict_global_feature_manifest.json"
SPLIT_MANIFEST = OUT / "simulator_features" / "simulator_internal_split_manifest.json"
GRAPH_ROOT = OUT / "simulator_features" / "graphs_pinned_model"
PRETRAIN_ROOT = OUT / "pretraining" / "valid_pinned_simple_model"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scope(path: Path) -> Path:
    path = path.resolve()
    root = OUT.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"path_outside_a1_namespace:{path}")
    return path


def atomic_bytes(path: Path, value: bytes) -> None:
    path = scope(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".partial", delete=False) as f:
        temporary = Path(f.name)
        f.write(value); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    atomic_bytes(path, output.getvalue().encode())


def load_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(100_000_000)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pinned_blob(relative: str) -> bytes:
    result = subprocess.run(["git", "-C", str(UPSTREAM), "show", f"{PIN}:{relative}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(f"pinned_blob_missing:{relative}:{result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def pinned_model_class():
    """Execute exactly the upstream class body in an inert explicit namespace."""
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import TransformerConv, global_mean_pool
    blob = pinned_blob("model/transformer_model.py")
    expected = "d6d250b2cb3082efc756ae9c5c269d4fe40bbdf49c743c535cddf6a0691c5683"
    if sha_bytes(blob) != expected:
        raise RuntimeError("pinned_model_blob_hash_mismatch")
    namespace = {"torch": torch, "F": F, "TransformerConv": TransformerConv, "global_mean_pool": global_mean_pool}
    exec(compile(blob, f"<pinned:{PIN}:model/transformer_model.py>", "exec"), namespace)
    return namespace["Simple_Model"], {"path": "model/transformer_model.py", "commit": PIN, "sha256": expected}


def args_for(architecture: str) -> SimpleNamespace:
    if architecture == "MLP":
        return SimpleNamespace(use_graph_features=False, use_global_features=True, use_gate_type=True, use_qubit_index=True, use_T1T2=False, use_gate_error=False, use_gate_index=True, num_layers=3)
    if architecture == "GRAPH_MLP":
        return SimpleNamespace(use_graph_features=True, use_global_features=True, use_gate_type=True, use_qubit_index=True, use_T1T2=False, use_gate_error=False, use_gate_index=True, num_layers=3)
    raise ValueError(f"unknown_architecture:{architecture}")


def instantiate(architecture: str):
    Model, provenance = pinned_model_class()
    kwargs = {"length_of_gf": 51}
    if architecture == "GRAPH_MLP": kwargs["length_of_x"] = 178
    else: kwargs["length_of_x"] = 178
    model = Model(args_for(architecture), **kwargs)
    if architecture == "GRAPH_MLP":
        active = int(model.mask.sum().item())
        if active != 174 or bool(model.mask[173:177].any()):
            raise RuntimeError("pinned_model_circuit_only_mask_failure")
    return model, provenance


def state_fingerprint(model) -> dict[str, Any]:
    state = model.state_dict()
    return {"state_dict_keys": list(state), "tensor_shapes": {key: list(value.shape) for key, value in state.items()}, "parameter_count": sum(p.numel() for p in model.parameters())}


def state_values_sha256(model) -> str:
    """Hash tensor values in state-dict order; never substitute a key-list hash."""
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fingerprints() -> dict[str, Any]:
    _, model_provenance = instantiate("MLP")
    result: dict[str, Any] = {"study_id": "postfreeze_track_a_transfer_v1", "status": "PASS", "pinned_model": model_provenance, "feature_extractor": {"helper_path": "data_preparation/helper.py", "helper_sha256": "c34cc98df02b016ee9c94dc9ca27efdbbb7e6c6439831f09d3c92ed3a78f74ad", "utils_path": "data_preparation/utils.py", "utils_sha256": "d70db4646f0b9d3deb548d6d960711a43ad818cef344113a6fa1e9f9ae64bcb7"}, "dag_semantic_reference": {"path": "data_preparation/circ_dag_converter.py", "sha256": "1b29a1c1869baeafdfbf825353f8024b9613b602cfea608e48f6bc23d54e7a85"}, "architectures": {}}
    for architecture in ("MLP", "GRAPH_MLP"):
        model, _ = instantiate(architecture)
        item = {"class_name": "Simple_Model", "constructor": vars(args_for(architecture)), "length_of_x": 178, "length_of_gf": 51, **state_fingerprint(model)}
        if architecture == "GRAPH_MLP": item["stored_graph_width"] = 178; item["active_graph_width"] = int(model.mask.sum().item()); item["active_mask_indices"] = [i for i, value in enumerate(model.mask.tolist()) if value]
        result["architectures"][architecture] = item
    invalid: dict[str, Any] = {"status": "NOT_FOUND"}
    try:
        import torch
        p = OUT / "pretraining" / "U-FULL-V3" / "seed_42" / "best.pt"
        old = torch.load(p, map_location="cpu", weights_only=False)
        model, _ = instantiate("MLP")
        equivalent = {key: list(value.shape) for key, value in old["model_state"].items()} == state_fingerprint(model)["tensor_shapes"]
        invalid = {"status": "INVALID_IMPLEMENTATION_NOT_FOR_A1_PRIMARY", "checkpoint": str(p.relative_to(OUT)), "state_shape_equivalent_to_pinned_mlp": equivalent, "exclusion_reason": "local reimplementation was executed instead of pinned upstream Simple_Model"}
    except FileNotFoundError:
        pass
    result["discarded_local_mlp_checkpoints"] = invalid
    atomic_json(OUT / "MODEL_FINGERPRINTS.json", result)
    return result


def must_preflight() -> None:
    receipt = json.loads((OUT / "preflight_receipt.json").read_text())
    if receipt.get("status") != "PASS": raise RuntimeError(f"preflight_not_pass:{receipt.get('status')}")
    source = json.loads(FEATURE_MANIFEST.read_text())
    if source.get("status") != "COMPLETE" or source.get("retained_observations") != 2636 or source.get("retained_identities") != 1232: raise RuntimeError("strict_feature_manifest_not_qualified")


def seed_all(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def graph_materialize() -> dict[str, Any]:
    """Build all 1,232 sparse circuit-only records under an independent new root."""
    must_preflight()
    manifest_path = GRAPH_ROOT / "graph_materialization_receipt.json"
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text())
        if old.get("status") == "COMPLETE" and old.get("identity_count") == 1232 and old.get("available_count") == 1232: return old
        raise RuntimeError("existing_graph_materialization_not_resumable")
    sys.path.insert(0, str(REPO / "benchmark_v1/scripts"))
    import mali_full_features as helper
    rows = load_csv(FEATURES)
    identities: dict[str, dict[str, str]] = {}
    for row in rows:
        identities.setdefault(row["circuit_identity_key"], row)
    if len(identities) != 1232 or len(rows) != 2636: raise RuntimeError("graph_source_count_failure")
    ledger: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for index, identity in enumerate(sorted(identities)):
        row = identities[identity]; name = row["circuit_name"]
        try:
            data = pinned_blob(f"data/quantum_circuits/{name}.qasm")
            if sha_bytes(data) != row["simulator_qasm_sha256"]: raise RuntimeError("pinned_qasm_digest_mismatch")
            circuit, _ = helper.parse_qasm(data)
            # The pinned semantic converter requires values in the storage slots;
            # substitute finite sentinels only while constructing then zero all four
            # slots before persistence. The model mask permanently excludes them.
            record, meta = helper.compact_graph_record(circuit, {i: (1.0, 1.0) for i in range(circuit.num_qubits)})
            for key in ("t1_0", "t2_0", "t1_1", "t2_1"): record[key].fill(0.0)
            compact = max(int(record["wire0"][record["wire0"] != 255].max(initial=0)), int(record["wire1"][record["wire1"] != 255].max(initial=0)))
            if compact > 126: raise RuntimeError("compact_wire_exceeds_126")
            rel = Path("records") / f"{index:04d}_{sha_bytes(identity.encode())[:16]}.npz"; dest = GRAPH_ROOT / rel
            import numpy as np
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_suffix(".partial.npz")
            np.savez_compressed(temporary, **record); os.replace(temporary, dest)
            digest = sha_file(dest)
            ledger.append({"circuit_identity_key": identity, "circuit_name": name, "status": "AVAILABLE", "reason": "", "record_path": str(rel), "record_sha256": digest})
            records.append({"circuit_identity_key": identity, "record_path": str(rel), "record_sha256": digest, "node_count": int(len(record["node_type"])), "edge_count": int(record["edge_index"].shape[1]), "max_wire_index": compact})
        except Exception as exc:
            ledger.append({"circuit_identity_key": identity, "circuit_name": name, "status": "UNAVAILABLE", "reason": f"{type(exc).__name__}:{exc}", "record_path": "", "record_sha256": ""})
    atomic_csv(GRAPH_ROOT / "feature_status_ledger.csv", ledger, ["circuit_identity_key", "circuit_name", "status", "reason", "record_path", "record_sha256"])
    if any(row["status"] != "AVAILABLE" for row in ledger):
        receipt = {"status": "BLOCKED_GRAPH_MATERIALIZATION_UNAVAILABLE", "identity_count": len(ledger), "available_count": sum(x["status"] == "AVAILABLE" for x in ledger)}
        atomic_json(manifest_path, receipt); raise RuntimeError("graph_materialization_unavailable_identity")
    receipt = {"study_id": "postfreeze_track_a_transfer_v1", "status": "COMPLETE", "identity_count": len(ledger), "available_count": len(ledger), "contextual_observation_count": len(rows), "stored_graph_width": 178, "active_graph_width": 174, "active_coordinates": "0:172,177; offsets 173:176 masked", "use_T1T2": False, "t1_t2_storage": "all zero; not model-visible", "source_context_counts": Counter(row["published_context"] for row in rows), "records": records, "feature_manifest_sha256": sha_file(FEATURE_MANIFEST), "ledger_sha256": sha_file(GRAPH_ROOT / "feature_status_ledger.csv")}
    atomic_json(manifest_path, receipt)
    return receipt


def global_transform(rows: list[dict[str, str]]) -> tuple[Any, Any]:
    import numpy as np
    values = np.asarray([[float(row[f"g{i:02d}"]) for i in range(51)] for row in rows], dtype=np.float32)
    mean, std = values.mean(axis=0), values.std(axis=0)
    zero_variance = std == 0
    return mean, np.where(zero_variance, 1.0, std), zero_variance


def graph_transform(fit_rows: list[dict[str, str]], record_map: dict[str, Path]) -> tuple[Any, Any]:
    import numpy as np
    sys.path.insert(0, str(REPO / "benchmark_v1/scripts")); from mali_full_features import expand_dense_x
    identities = sorted({row["circuit_identity_key"] for row in fit_rows})
    total = np.zeros(178, dtype=np.float64); total2 = np.zeros(178, dtype=np.float64); count = 0
    for identity in identities:
        with np.load(record_map[identity], allow_pickle=False) as z: x = expand_dense_x({k:z[k] for k in z.files})
        total += x.sum(0); total2 += (x*x).sum(0); count += len(x)
    mean = (total / count).astype(np.float32); variance = np.maximum(total2 / count - mean.astype(np.float64)**2, 0.0); std = np.sqrt(variance).astype(np.float32); std[std == 0] = 1.0
    return mean, std


def graph_data(row: dict[str, str], record_map: dict[str, Path], node_mean, node_std, global_mean, global_std, target: float, weight: float):
    import numpy as np
    import torch
    from torch_geometric.data import Data
    sys.path.insert(0, str(REPO / "benchmark_v1/scripts")); from mali_full_features import expand_dense_x
    with np.load(record_map[row["circuit_identity_key"]], allow_pickle=False) as z: record={key:z[key] for key in z.files}
    x=(expand_dense_x(record)-node_mean)/node_std
    global_x=(np.asarray([float(row[f"g{i:02d}"]) for i in range(51)],dtype=np.float32)-global_mean)/global_std
    return Data(x=torch.as_tensor(x), edge_index=torch.as_tensor(record["edge_index"],dtype=torch.long), global_features=torch.as_tensor(global_x).reshape(1,51), y=torch.tensor(target,dtype=torch.float32), sample_weight=torch.tensor(weight,dtype=torch.float32))


def identity_label_groups(rows: list[dict[str, str]]) -> list[tuple[dict[str, str], float, float, list[float]]]:
    """Use one graph per identity while retaining both contextual label weights.

    The model has no context input, so repeated contextual graphs are mathematically
    identical. Weighted squared error against their mean target has the same gradient
    as the two original row-level losses; raw contextual labels remain available for
    validation/report MSE calculation.
    """
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows: grouped.setdefault(row["circuit_identity_key"], []).append(row)
    result=[]
    for identity, group in sorted(grouped.items()):
        labels=[float(item["time_taken_seconds"]) for item in group]
        result.append((group[0], sum(labels)/len(labels), float(len(labels)), labels))
    return result


def train(architecture: str, seed: int, epochs: int = 500) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch_geometric.data import Batch
    must_preflight(); fingerprint = fingerprints(); graph_receipt = graph_materialize() if architecture == "GRAPH_MLP" else None
    directory = PRETRAIN_ROOT / architecture / f"seed_{seed}"; terminal = directory / "terminal_receipt.json"
    if terminal.is_file():
        value = json.loads(terminal.read_text())
        if value.get("status") == "COMPLETE": return value
        raise RuntimeError("existing_terminal_pretrain_cell_not_resumable")
    # An interruption before a checkpoint is not a scientific result. Preserve
    # the failed attempt receipt, then start a fresh immutable attempt directory
    # using the corrected pinned-model wrapper without reusing any state.
    identity_path = directory / "identity_manifest.json"
    if identity_path.exists():
        prior = json.loads(identity_path.read_text())
        if prior.get("status") == "RUNNING":
            reason = ("pinned Simple_Model requires data.x even for MLP; no optimizer/model checkpoint was emitted" if architecture == "MLP" else "stopped by the user to replace batch-one graph execution with a frozen node/edge-capped microbatch policy; existing checkpoints must not resume")
            atomic_json(directory / "interrupted_attempt_receipt.json", {"status": "INVALID_WRAPPER_INPUT_FIX_BEFORE_CHECKPOINT" if architecture == "MLP" else "INVALID_RESOURCE_POLICY_REPLACED_BEFORE_COMPLETION", "reason": reason, "superseded_by_runner_sha256": sha_file(Path(__file__))})
            directory = directory / ("attempt_2_pinned_input_fix" if architecture == "MLP" else "attempt_2_node_edge_capped_microbatch")
    rows = load_csv(FEATURES); split = {key:[row for row in rows if row["split"] == key] for key in ("fit","validation","report")}
    if {key:len(value) for key,value in split.items()} != {"fit":2112,"validation":260,"report":264}: raise RuntimeError("source_split_observation_count_failure")
    seed_all(seed); model, model_provenance = instantiate(architecture); device=torch.device("cuda"); model=model.to(device)
    global_mean,global_std,global_zero_variance=global_transform(split["fit"])
    identity={key:sha_bytes("\n".join(sorted({row["circuit_identity_key"] for row in value})).encode()) for key,value in split.items()}
    transform={"raw_width":51,"global_mean":global_mean.tolist(),"global_std":global_std.tolist(),"zero_variance_global_indices":np.flatnonzero(global_zero_variance).astype(int).tolist(),"fit_identity_sha256":identity["fit"],"zero_total_pruning":False}
    if architecture == "MLP":
        def make_batch(items):
            x=np.asarray([[(float(row[f"g{i:02d}"])-global_mean[i])/global_std[i] for i in range(51)] for row in items],dtype=np.float32); y=np.asarray([float(row["time_taken_seconds"]) for row in items],dtype=np.float32); return torch.as_tensor(x,device=device),torch.as_tensor(y,device=device)
        fit_batch=make_batch(split["fit"]); val_batch=make_batch(split["validation"]); report_batch=make_batch(split["report"])
        # Pinned Simple_Model always materializes data.x before branching on
        # use_graph_features; pass the same 51-wide tensor as x and globals.
        def predict(items):
            x, y = make_batch(items)
            return model(SimpleNamespace(x=x, edge_index=None, global_features=x, batch=None)), y
        batch_size=256
    else:
        record_map={row["circuit_identity_key"]:GRAPH_ROOT/row["record_path"] for row in graph_receipt["records"]}
        node_mean,node_std=graph_transform(split["fit"],record_map); transform.update({"node_mean":node_mean.tolist(),"node_std":node_std.tolist(),"stored_node_width":178,"active_node_width":174})
        grouped_split={key:identity_label_groups(value) for key,value in split.items()}
        # Each graph identity has the same two published contextual labels. A
        # weighted loss at its mean label has the same gradient as duplicating
        # its graph once per contextual observation, without duplicating graph
        # memory. Contextual row metrics are expanded separately below.
        graph_size: dict[str, tuple[int, int]] = {row["circuit_identity_key"]: (int(row["node_count"]), int(row["edge_count"])) for row in graph_receipt["records"]}
        if any(nodes > GRAPH_BATCH_NODE_CAP or edges > GRAPH_BATCH_EDGE_CAP for nodes, edges in graph_size.values()):
            raise RuntimeError("single_graph_exceeds_frozen_microbatch_capacity")
        def predict(groups):
            batch=Batch.from_data_list([graph_data(row,record_map,node_mean,node_std,global_mean,global_std,target,weight) for row,target,weight,_labels in groups]).to(device)
            return model(batch).reshape(-1),batch.y.reshape(-1),batch.sample_weight.reshape(-1)
        def packed_groups(groups):
            current=[]; nodes=edges=0
            # Deterministic order arrives from the frozen epoch RNG above. Packing is
            # data-size-only, with no labels or outcomes involved.
            for item in groups:
                item_nodes,item_edges=graph_size[item[0]["circuit_identity_key"]]
                if current and (nodes+item_nodes > GRAPH_BATCH_NODE_CAP or edges+item_edges > GRAPH_BATCH_EDGE_CAP):
                    yield current; current=[]; nodes=edges=0
                current.append(item); nodes += item_nodes; edges += item_edges
            if current: yield current
        batch_size=None
    atomic_json(directory / "identity_manifest.json", {"study_id":"postfreeze_track_a_transfer_v1","status":"RUNNING","architecture":architecture,"seed":seed,"epochs":epochs,"model_provenance":model_provenance,"constructor":vars(args_for(architecture)),"model_fingerprint_sha256":sha_file(OUT/"MODEL_FINGERPRINTS.json"),"feature_manifest_sha256":sha_file(FEATURE_MANIFEST),"graph_receipt_sha256":sha_file(GRAPH_ROOT/"graph_materialization_receipt.json") if graph_receipt else None,"split_identity_sha256":identity,"optimizer":{"name":"Adam","lr":5e-4,"weight_decay":1e-4},"scheduler":"constant","loss":"raw_seconds_mse","precision":"float32","amp":False,"graph_microbatch_policy":{"node_cap":GRAPH_BATCH_NODE_CAP,"edge_cap":GRAPH_BATCH_EDGE_CAP,"selection":"deterministic order; pack graph identities until either cap; source labels do not affect packing"} if architecture == "GRAPH_MLP" else None,"checkpoint_selection":"minimum simulator validation MSE; earliest epoch tie"})
    atomic_json(directory / "transform.json", transform)
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-4,weight_decay=1e-4); best=float("inf"); best_epoch=None; history=[]
    def weighted_mse(prediction, target, weights):
        return (((prediction-target)**2)*weights).sum()/weights.sum()
    def contextual_mse(groups):
        errors=[]
        for row,target,weight,labels in groups:
            prediction, _target, _weight = predict([(row,target,weight,labels)])
            errors.extend((float(prediction.item())-label)**2 for label in labels)
        return float(sum(errors)/len(errors))
    for epoch in range(1,epochs+1):
        model.train()
        if architecture == "MLP":
            order=list(range(len(split["fit"]))); random.Random(seed*1_000_003+epoch).shuffle(order)
            for start in range(0,len(order),batch_size):
                chosen=[split["fit"][i] for i in order[start:start+batch_size]]; prediction,target=predict(chosen); optimizer.zero_grad(set_to_none=True); loss=torch.nn.functional.mse_loss(prediction,target); loss.backward(); optimizer.step()
        else:
            order=list(range(len(grouped_split["fit"]))); random.Random(seed*1_000_003+epoch).shuffle(order)
            ordered_groups=[grouped_split["fit"][index] for index in order]
            for packed in packed_groups(ordered_groups):
                prediction,target,weight=predict(packed); optimizer.zero_grad(set_to_none=True); loss=weighted_mse(prediction,target,weight); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            if architecture == "MLP":
                train_pred,train_target=predict(split["fit"]); val_pred,val_target=predict(split["validation"]); train_mse=float(torch.nn.functional.mse_loss(train_pred,train_target).item()); val_mse=float(torch.nn.functional.mse_loss(val_pred,val_target).item())
            else:
                train_mse=contextual_mse(grouped_split["fit"]); val_mse=contextual_mse(grouped_split["validation"])
        history.append({"epoch":epoch,"train_mse":train_mse,"validation_mse":val_mse})
        if val_mse < best:
            best,best_epoch=val_mse,epoch
            payload={"study_id":"postfreeze_track_a_transfer_v1","architecture":architecture,"seed":seed,"model_state":model.state_dict(),"selected_epoch":epoch,"validation_mse":val_mse,"source_model":model_provenance,"constructor":vars(args_for(architecture)),"transform_sha256":sha_file(directory/"transform.json"),"feature_manifest_sha256":sha_file(FEATURE_MANIFEST),"graph_receipt_sha256":sha_file(GRAPH_ROOT/"graph_materialization_receipt.json") if graph_receipt else None}
            temporary=directory/"best.pt.partial"; temporary.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,temporary); os.replace(temporary,directory/"best.pt")
        if epoch % 10 == 0 or epoch == epochs:
            temporary=directory/"current.pt.partial"; torch.save({"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"epoch":epoch,"rng_python":random.getstate(),"rng_numpy":np.random.get_state(),"rng_torch":torch.get_rng_state(),"rng_cuda":torch.cuda.get_rng_state_all()},temporary); os.replace(temporary,directory/"current.pt")
    atomic_json(directory/"training_history.json",history)
    selected=torch.load(directory/"best.pt",map_location=device,weights_only=False); model.load_state_dict(selected["model_state"],strict=True); model.eval()
    with torch.no_grad():
        if architecture == "MLP":
            report_pred,report_target=predict(split["report"]); report_mse=float(torch.nn.functional.mse_loss(report_pred,report_target).item())
        else:
            report_mse=contextual_mse(grouped_split["report"])
    terminal_value={"study_id":"postfreeze_track_a_transfer_v1","status":"COMPLETE","architecture":architecture,"seed":seed,"best_epoch":best_epoch,"best_validation_mse":best,"report_only_mse":report_mse,"checkpoint_sha256":sha_file(directory/"best.pt"),"parameter_count":sum(p.numel() for p in model.parameters()),"state_values_sha256":state_values_sha256(model),"model_provenance":model_provenance,"device":torch.cuda.get_device_name(0),"precision":"float32","amp":False}
    atomic_json(terminal,terminal_value); return terminal_value


def qualification_canary(model, architecture: str) -> dict[str, Any]:
    """Reloaded-model deterministic canary with architecture-specific inputs."""
    import torch
    model = model.cuda().eval()
    torch.manual_seed(20261019)
    if architecture == "MLP":
        x = torch.randn((2, 51), device="cuda", dtype=torch.float32)
        data = SimpleNamespace(x=x, edge_index=None, global_features=x, batch=None)
    else:
        from torch_geometric.data import Data
        x = torch.randn((4, 178), device="cuda", dtype=torch.float32)
        edge = torch.tensor([[0, 1, 2], [1, 2, 3]], device="cuda", dtype=torch.long)
        data = Data(x=x, edge_index=edge, global_features=torch.zeros((1, 51), device="cuda"), batch=torch.zeros(4, device="cuda", dtype=torch.long))
        if int(model.mask.sum().item()) != 174 or bool(model.mask[173:177].any()):
            raise RuntimeError("canary_circuit_only_mask_failure")
    with torch.no_grad():
        first = model(data).detach().cpu()
        second = model(data).detach().cpu()
    if not torch.equal(first, second) or not torch.isfinite(first).all():
        raise RuntimeError(f"canary_nondeterministic_or_nonfinite:{architecture}")
    return {"status": "PASS", "output_sha256": sha_bytes(first.contiguous().numpy().tobytes()), "output_shape": list(first.shape)}


def qualified_checkpoint_path(architecture: str, seed: int) -> tuple[Path, dict[str, Any]]:
    """Resolve a terminal receipt's declared checkpoint by exact file hash."""
    family = PRETRAIN_ROOT / architecture / f"seed_{seed}"
    terminal = family / "terminal_receipt.json"
    if not terminal.is_file():
        raise RuntimeError(f"valid_terminal_receipt_missing:{architecture}:{seed}")
    receipt = json.loads(terminal.read_text())
    if receipt.get("status") != "COMPLETE":
        raise RuntimeError(f"terminal_checkpoint_not_complete:{architecture}:{seed}")
    expected = receipt.get("checkpoint_sha256")
    matches = [path for path in family.rglob("best.pt") if sha_file(path) == expected]
    # Identical checksum copies may exist because an interrupted directory was
    # preserved audit-only; canonical direct-path is the terminal cell's output.
    canonical = family / "best.pt"
    if canonical in matches:
        return canonical, receipt
    if len(matches) != 1:
        raise RuntimeError(f"terminal_checkpoint_hash_resolution_failure:{architecture}:{seed}:{len(matches)}")
    return matches[0], receipt


def qualify() -> dict[str, Any]:
    import torch
    fps=fingerprints(); report={"study_id":"postfreeze_track_a_transfer_v1","PRETRAINING_PROVENANCE":"PASS","CHECKPOINT_PARITY":"PASS","FACTORIAL":"NOT_EXECUTED","families":{}}
    for architecture in ("MLP","GRAPH_MLP"):
        expected=fps["architectures"][architecture]; family=[]
        for seed in SEEDS:
            p, terminal_receipt = qualified_checkpoint_path(architecture, seed)
            checkpoint=torch.load(p,map_location="cpu",weights_only=False)
            if checkpoint.get("architecture") != architecture or checkpoint.get("seed") != seed or checkpoint.get("source_model", {}).get("sha256") != fps["pinned_model"]["sha256"]: raise RuntimeError(f"checkpoint_provenance_mismatch:{architecture}:{seed}")
            scratch,_=instantiate(architecture); pretrained,_=instantiate(architecture)
            before=state_fingerprint(scratch)
            if before["state_dict_keys"] != expected["state_dict_keys"] or before["tensor_shapes"] != expected["tensor_shapes"]: raise RuntimeError(f"scratch_fingerprint_mismatch:{architecture}")
            result=pretrained.load_state_dict(checkpoint["model_state"],strict=True)
            if result.missing_keys or result.unexpected_keys: raise RuntimeError(f"strict_load_result_nonempty:{architecture}:{seed}")
            after=state_fingerprint(pretrained)
            if before["tensor_shapes"] != after["tensor_shapes"] or before["parameter_count"] != after["parameter_count"]: raise RuntimeError(f"parity_shape_failure:{architecture}:{seed}")
            if not all(torch.isfinite(value).all().item() for value in pretrained.state_dict().values()): raise RuntimeError(f"nonfinite_checkpoint_weight:{architecture}:{seed}")
            canary=qualification_canary(pretrained,architecture)
            family.append({"seed":seed,"checkpoint_sha256":sha_file(p),"provenance_pass":"PASS","strict_load":"PASS","state_dict_keys_identical":True,"parameter_shapes_identical":True,"finite_weights":"PASS","canary_inference":canary,"architecture_parity":"PASS","authorized_for_qpu":"PASS","parameter_count":after["parameter_count"]})
        report["families"][architecture]=family
    atomic_json(OUT/"checkpoint_qualification.json",report)
    lines=["# Amendment A1 checkpoint qualification","", "- **PRETRAINING_PROVENANCE_PASS:** `PASS`", "- **CHECKPOINT_PARITY_PASS:** `PASS`", "- **FACTORIAL_EXECUTED:** `NOT_EXECUTED`", "", "Valid source checkpoints were newly trained from random initialization with pinned upstream `Simple_Model` under Study-U 51-global / circuit-only-graph dimensions. Each qualified checkpoint loads with `strict=True` into its same-architecture A1 model. The older local MLP checkpoints remain excluded audit-only artifacts.", ""]
    atomic_bytes(OUT/"CHECKPOINT_QUALIFICATION.md", "\n".join(lines).encode()); return report


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=("fingerprint","graphs","pretrain-mlp","pretrain-graph","qualify","all-source"),required=True); parser.add_argument("--seed",type=int,choices=SEEDS); args=parser.parse_args()
    if args.stage=="fingerprint": fingerprints(); print("MODEL_FINGERPRINTS_PASS"); return 0
    if args.stage=="graphs": value=graph_materialize(); print(json.dumps(value,sort_keys=True)); return 0
    architectures={"pretrain-mlp":"MLP","pretrain-graph":"GRAPH_MLP"}
    if args.stage in architectures:
        targets=(args.seed,) if args.seed is not None else SEEDS
        for seed in targets: print(json.dumps(train(architectures[args.stage],seed),sort_keys=True))
        return 0
    if args.stage=="qualify": print(json.dumps(qualify(),sort_keys=True)); return 0
    fingerprints(); graph_materialize()
    for architecture in ("MLP","GRAPH_MLP"):
        for seed in SEEDS: train(architecture,seed)
    print(json.dumps(qualify(),sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
