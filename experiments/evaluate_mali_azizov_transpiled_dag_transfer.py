#!/usr/bin/env python3
"""Ma-Li two-stage estimator on transpiled native DAGs (Azizov circuit object).

Simulator graphs: FakeWashingtonV2/FakeSherbrooke native DAGs.
QPU graphs: existing FakeOsaka/FakeKyoto native DAGs.
The GNN sees coarsened topological bins of the mapped circuit, with physical
T1/T2, not the logical QASM DAG and not scalar T1/T2-free features alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from mali_family_ood_common import affine_map, component_families, filename_family, metrics
from mali_transpiled_dag_common import (
    GLOBAL_DIM,
    MAX_BINS,
    NODE_DIM,
    coarsen_graph,
    load_snapshot_map,
    read_qubit_t1_t2,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def hashed_folds(keys: list[str], folds: int, seed: int = 1234) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    unique = sorted(set(keys))
    rng.shuffle(unique)
    return {key: (index % folds) + 1 for index, key in enumerate(unique)}


class CoarsenedDagGNN(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3) -> None:
        super().__init__()
        self.node_in = nn.Linear(NODE_DIM + 2, hidden)  # + backend dummy pair
        self.layers = nn.ModuleList([nn.Linear(hidden * 2, hidden) for _ in range(layers)])
        self.gf = nn.Sequential(nn.Linear(GLOBAL_DIM + 2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, nodes, edges, weights, global_x, backend_dummy):
        dummy = backend_dummy.expand(nodes.size(0), -1)
        hidden = torch.nn.functional.gelu(self.node_in(torch.cat([nodes, dummy], dim=-1)))
        n = hidden.size(0)
        has_edges = bool(edges.numel())
        src = edges[0] if has_edges else None
        dst = edges[1] if has_edges else None
        for conv in self.layers:
            if has_edges:
                messages = hidden[src] * weights[:, None]
                agg = hidden.new_zeros(n, hidden.size(-1))
                agg.index_add_(0, dst, messages)
                deg = hidden.new_zeros(n, 1)
                deg.index_add_(0, dst, weights[:, None])
                neigh = agg / deg.clamp(min=1.0)
            else:
                neigh = hidden.new_zeros(hidden.shape)
            hidden = torch.nn.functional.gelu(hidden + conv(torch.cat([hidden, neigh], dim=-1)))
        pooled = hidden.mean(dim=0)
        global_h = self.gf(torch.cat([global_x, backend_dummy], dim=-1))
        return self.head(torch.cat([pooled, global_h], dim=-1)).squeeze()


def backend_dummy(name: str, device: torch.device) -> torch.Tensor:
    # [is_sherbrooke_or_kyoto, is_qpu]
    qpu = 1.0 if name in {"osaka", "kyoto"} else 0.0
    second = 1.0 if name in {"sherbrooke", "kyoto"} else 0.0
    return torch.tensor([second, qpu], dtype=torch.float32, device=device)


def pad_nodes(nodes: np.ndarray) -> np.ndarray:
    if nodes.shape[0] >= MAX_BINS:
        return nodes[:MAX_BINS]
    pad = np.zeros((MAX_BINS - nodes.shape[0], nodes.shape[1]), dtype=np.float32)
    return np.vstack([nodes, pad])


def load_example(graph_path: Path, t1: np.ndarray, t2: np.ndarray, cache_dir: Path) -> dict:
    cache = cache_dir / f"{graph_path.stem}.npz"
    if cache.exists():
        with np.load(cache) as archive:
            return {
                "nodes": archive["nodes"],
                "edges": archive["edges"],
                "weights": archive["weights"],
                "global_x": archive["global_x"],
            }
    nodes, edges, weights, global_x = coarsen_graph(graph_path, t1, t2)
    # remap edges after padding is not needed if we keep original bin count
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, nodes=nodes, edges=edges, weights=weights, global_x=global_x)
    return {"nodes": nodes, "edges": edges, "weights": weights, "global_x": global_x}


def feature_stats(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes = np.concatenate([row["_example"]["nodes"] for row in rows], axis=0)
    globals_ = np.stack([row["_example"]["global_x"] for row in rows], axis=0)
    return nodes.mean(axis=0), nodes.std(axis=0) + 1e-6, globals_.mean(axis=0), globals_.std(axis=0) + 1e-6


def materialize(rows: list[dict], stats, device: torch.device) -> None:
    node_mean, node_std, gf_mean, gf_std = stats
    for row in rows:
        example = row["_example"]
        nodes = (example["nodes"] - node_mean) / node_std
        global_x = (example["global_x"] - gf_mean) / gf_std
        row["_tensor"] = to_torch(
            {"nodes": nodes, "edges": example["edges"], "weights": example["weights"], "global_x": global_x},
            row["backend"],
            device,
        )


def to_torch(example: dict, backend: str, device: torch.device) -> dict:
    return {
        "nodes": torch.from_numpy(np.asarray(example["nodes"], dtype=np.float32)).to(device),
        "edges": torch.from_numpy(example["edges"]).to(device),
        "weights": torch.from_numpy(example["weights"]).to(device),
        "global_x": torch.from_numpy(np.asarray(example["global_x"], dtype=np.float32)).to(device),
        "dummy": backend_dummy(backend, device),
    }


@torch.no_grad()
def predict_rows(model, rows, device) -> np.ndarray:
    model.eval()
    out = []
    for row in rows:
        item = row["_tensor"]
        pred = model(item["nodes"], item["edges"], item["weights"], item["global_x"], item["dummy"])
        out.append(float(decode_seconds(pred).cpu()))
    return np.asarray(out, dtype=np.float64)


def decode_seconds(log_pred: torch.Tensor) -> torch.Tensor:
    return torch.expm1(log_pred.clamp(-1.0, 8.0)).clamp(min=0.0)


def train_model(train_rows, val_rows, device, epochs: int, lr: float, seed: int, init=None) -> CoarsenedDagGNN:
    seed_all(seed)
    model = CoarsenedDagGNN().to(device)
    if init is not None:
        model.load_state_dict(init)
    y_train = np.asarray([row["target_seconds"] for row in train_rows], dtype=np.float64)
    median = float(np.median(y_train)) if len(y_train) else 1.0
    if init is None:
        with torch.no_grad():
            model.head[-1].bias.fill_(np.log1p(median))
            model.head[-1].weight.zero_()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(train_rows))
        total = 0.0
        opt.zero_grad(set_to_none=True)
        for step, index in enumerate(order, start=1):
            row = train_rows[int(index)]
            item = row["_tensor"]
            pred = model(item["nodes"], item["edges"], item["weights"], item["global_x"], item["dummy"])
            target = torch.log1p(torch.tensor(row["target_seconds"], dtype=torch.float32, device=device))
            loss = torch.nn.functional.smooth_l1_loss(pred, target)
            (loss / 8.0).backward()
            total += float(loss.item())
            if step % 8 == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        if val_rows:
            pred = np.log1p(predict_rows(model, val_rows, device))
            yv = np.log1p(np.asarray([row["target_seconds"] for row in val_rows], dtype=np.float64))
            val = float(np.mean((pred - yv) ** 2))
        else:
            val = total / max(len(train_rows), 1)
        if val < best_val:
            best_val = val
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def attach_graphs(rows, graph_lookup, t1t2, cache_dir, device):
    attached = []
    missing = 0
    for row in rows:
        if (len(attached) + missing) % 100 == 0:
            print(
                f"[attach] {cache_dir.name}: {len(attached)+missing}/{len(rows)} attached={len(attached)} missing={missing}",
                flush=True,
            )
        key = (row["qasm_sha256"], row["backend"])
        graph_path = graph_lookup.get(key)
        if graph_path is None or not Path(graph_path).exists():
            missing += 1
            continue
        t1, t2 = t1t2[row["backend"]]
        example = load_example(Path(graph_path), t1, t2, cache_dir)
        item = dict(row)
        item["_example"] = example
        attached.append(item)
    return attached, missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mali-root", type=Path, default=ROOT.parent / "Quantum-Execution-Time-Prediction")
    parser.add_argument("--ws-records", type=Path, default=ROOT / "artifacts/validation/mali_azizov_transpiled_dag_v1/ws_graph_records.csv")
    parser.add_argument("--qpu-manifest", type=Path, default=ROOT / "artifacts/validation/mali_physical_dag_v1/manifest.csv")
    parser.add_argument("--qpu-graphs", type=Path, default=ROOT / "work/mali_physical_dag_v1/graphs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/validation/mali_azizov_transpiled_dag_v1")
    parser.add_argument("--epochs-sim", type=int, default=40)
    parser.add_argument("--epochs-qpu", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--qpu-scratch-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = args.output_dir / "coarsened"
    snapshot_map = load_snapshot_map(
        [
            ROOT / "work/mali_azizov_transpiled_dag_v1/backend",
            ROOT / "work/mali_physical_dag_v1/backend",
        ]
    )
    t1t2 = {}
    for name in ("washington", "sherbrooke", "osaka", "kyoto"):
        path = snapshot_map.get(name)
        if path is None:
            continue
        t1t2[name] = read_qubit_t1_t2(path)

    split_rows = read_csv(ROOT / "artifacts/validation/mali_deep_protocol_v1/split_audit/row_manifest.csv")
    qpu_manifest = { (row["qasm_sha256"], row["backend"]): row for row in read_csv(args.qpu_manifest) }
    qpu_graph_lookup = {
        (row["qasm_sha256"], row["backend"]): str(args.qpu_graphs / f"{row['graph_id']}.npz")
        for row in read_csv(args.qpu_manifest)
    }
    qpu_rows = []
    for split in split_rows:
        key = (split["qasm_sha256"], split["backend"])
        graph_row = qpu_manifest[key]
        qpu_rows.append(
            {
                "circuit": split["circuit"],
                "backend": split["backend"],
                "qasm_sha256": split["qasm_sha256"],
                "filename_family": split["filename_family"],
                "family_component": split["family_component"],
                "group_fold": int(split["group_fold"]),
                "target_seconds": float(split["target_seconds"]),
                "graph_id": graph_row["graph_id"],
            }
        )
    qpu_rows, qpu_missing = attach_graphs(qpu_rows, qpu_graph_lookup, t1t2, cache_dir / "qpu", device)
    if qpu_missing:
        raise RuntimeError(f"missing {qpu_missing} QPU native DAGs")
    predictions = []
    summary = {"device": str(device), "n_qpu": len(qpu_rows), "qpu_missing": qpu_missing}

    # QPU scratch on grouped folds: ceiling for compiled/transpiled DAG.
    for fold in range(1, 6):
        print(f"[qpu scratch] fold {fold}/5 n_train={sum(1 for row in qpu_rows if row['group_fold'] != fold)}", flush=True)
        train = [row for row in qpu_rows if row["group_fold"] != fold]
        test = [row for row in qpu_rows if row["group_fold"] == fold]
        val = train[::5] or train
        fit = [row for index, row in enumerate(train) if index % 5]
        if not fit:
            fit = train
        materialize(fit + val + test, feature_stats(fit), device)
        model = train_model(fit, val, device, args.epochs_qpu, 1e-3, args.seed + fold)
        pred = predict_rows(model, test, device)
        for row, value in zip(test, pred):
            predictions.append(
                {
                    "split": f"qpu_scratch_fold_{fold}",
                    "model": "qpu_scratch_transpiled_dag",
                    "circuit": row["circuit"],
                    "backend": row["backend"],
                    "filename_family": row["filename_family"],
                    "family_component": row["family_component"],
                    "qasm_sha256": row["qasm_sha256"],
                    "target_seconds": row["target_seconds"],
                    "prediction_seconds": value,
                }
            )

    sim_rows = []
    if args.ws_records.is_file() and not args.qpu_scratch_only:
        labels = {}
        for device_name in ("washington", "sherbrooke"):
            path = args.mali_root / "data" / f"{device_name}_time_taken.csv"
            for row in read_csv(path):
                name = row.get("circuit_name") or row.get("quantum_circuit")
                labels[(name, device_name)] = float(row["time_taken"])
        ws_lookup = {}
        for row in read_csv(args.ws_records):
            if row.get("status") not in {"ok", "built", "cached"}:
                continue
            ws_lookup[(row["qasm_sha256"], row["backend"])] = row["graph_path"]
            if (row["circuit"], row["backend"]) not in labels:
                continue
            sim_rows.append(
                {
                    "circuit": row["circuit"],
                    "backend": row["backend"],
                    "qasm_sha256": row["qasm_sha256"],
                    "filename_family": filename_family(row["circuit"]),
                    "target_seconds": labels[(row["circuit"], row["backend"])],
                    "priority": row.get("priority"),
                }
            )
        sim_rows, sim_missing = attach_graphs(sim_rows, ws_lookup, t1t2, cache_dir / "sim", device)
        summary.update({"n_sim": len(sim_rows), "sim_missing_graphs": sim_missing})
        if len(sim_rows) >= 50:
            folds = hashed_folds([row["qasm_sha256"] for row in sim_rows], 5, args.seed)
            for fold in range(1, 6):
                print(f"[sim grouped] fold {fold}/5 n={sum(1 for row in sim_rows if folds[row['qasm_sha256']] != fold)}", flush=True)
                train = [row for row in sim_rows if folds[row["qasm_sha256"]] != fold]
                test = [row for row in sim_rows if folds[row["qasm_sha256"]] == fold]
                val = train[::5] or train
                fit = [row for index, row in enumerate(train) if index % 5]
                if not fit:
                    fit = train
                materialize(fit + val + test, feature_stats(fit), device)
                model = train_model(fit, val, device, args.epochs_sim, 1e-3, args.seed + 20 + fold)
                pred = predict_rows(model, test, device)
                for row, value in zip(test, pred):
                    predictions.append(
                        {
                            "split": f"sim_grouped_fold_{fold}",
                            "model": "sim_transpiled_dag",
                            "circuit": row["circuit"],
                            "backend": row["backend"],
                            "filename_family": row["filename_family"],
                            "qasm_sha256": row["qasm_sha256"],
                            "target_seconds": row["target_seconds"],
                            "prediction_seconds": value,
                        }
                    )

            # Same-circuit transfer: train on all sim, affine on QPU folds.
            val = sim_rows[::8] or sim_rows
            fit = [row for index, row in enumerate(sim_rows) if index % 8] or sim_rows
            transfer_stats = feature_stats(fit)
            print(f"[transfer] train sim model n_fit={len(fit)} n_val={len(val)}", flush=True)
            materialize(fit + val, transfer_stats, device)
            sim_model = train_model(fit, val, device, args.epochs_sim, 1e-3, args.seed + 99)
            for fold in range(1, 6):
                print(f"[transfer] QPU fold {fold}/5", flush=True)
                qpu_train = [row for row in qpu_rows if row["group_fold"] != fold]
                qpu_test = [row for row in qpu_rows if row["group_fold"] == fold]
                materialize(qpu_train + qpu_test, transfer_stats, device)
                train_pred = predict_rows(sim_model, qpu_train, device)
                test_pred = predict_rows(sim_model, qpu_test, device)
                y_train = np.asarray([row["target_seconds"] for row in qpu_train], dtype=np.float64)
                mapped = affine_map(train_pred, y_train, test_pred)
                for row, value in zip(qpu_test, mapped):
                    predictions.append(
                        {
                            "split": f"transfer_incl_fold_{fold}",
                            "model": "sim_transpiled_dag_affine_qpu",
                            "circuit": row["circuit"],
                            "backend": row["backend"],
                            "filename_family": row["filename_family"],
                            "family_component": row["family_component"],
                            "qasm_sha256": row["qasm_sha256"],
                            "target_seconds": row["target_seconds"],
                            "prediction_seconds": float(value),
                        }
                    )
                ft_val = qpu_train[::5] or qpu_train
                ft_fit = [row for index, row in enumerate(qpu_train) if index % 5] or qpu_train
                ft_model = train_model(
                    ft_fit,
                    ft_val,
                    device,
                    max(10, args.epochs_qpu // 2),
                    3e-4,
                    args.seed + 200 + fold,
                    init=sim_model.state_dict(),
                )
                ft_pred = predict_rows(ft_model, qpu_test, device)
                for row, value in zip(qpu_test, ft_pred):
                    predictions.append(
                        {
                            "split": f"transfer_incl_fold_{fold}",
                            "model": "sim_transpiled_dag_finetune_qpu",
                            "circuit": row["circuit"],
                            "backend": row["backend"],
                            "filename_family": row["filename_family"],
                            "family_component": row["family_component"],
                            "qasm_sha256": row["qasm_sha256"],
                            "target_seconds": row["target_seconds"],
                            "prediction_seconds": float(value),
                        }
                    )

    def pooled(model_name: str) -> dict:
        selected = [row for row in predictions if row["model"] == model_name]
        if not selected:
            return {"n": 0}
        score = metrics(
            np.asarray([row["target_seconds"] for row in selected], dtype=float),
            np.asarray([row["prediction_seconds"] for row in selected], dtype=float),
        )
        score["n"] = len(selected)
        return score

    summary["metrics"] = {
        name: pooled(name)
        for name in (
            "qpu_scratch_transpiled_dag",
            "sim_transpiled_dag",
            "sim_transpiled_dag_affine_qpu",
            "sim_transpiled_dag_finetune_qpu",
        )
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "predictions.csv", predictions)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Ma–Li flow on transpiled native DAGs",
        "",
        "**Finding recorded:** 2026-09-22.",
        "",
        "The circuit object is `transpile(logical QASM, target backend)`.",
        "The GNN reads a 256-bin coarsening of that native DAG, including",
        "physical T1/T2. It is not Ma–Li's logical DAG and not scalar T1/T2-free features.",
        "",
        f"- device: `{device}`",
        f"- QPU rows: {len(qpu_rows)}",
        f"- simulator native-DAG rows attached: {summary.get('n_sim', 0)}",
        "",
        "| Model | log-R² / MAE |",
        "|---|---|",
    ]
    for name, title in (
        ("qpu_scratch_transpiled_dag", "QPU scratch, transpiled DAG"),
        ("sim_transpiled_dag", "Simulator in-domain, transpiled DAG"),
        ("sim_transpiled_dag_affine_qpu", "Sim DAG → affine QPU"),
        ("sim_transpiled_dag_finetune_qpu", "Sim DAG → fine-tune QPU"),
    ):
        score = summary["metrics"][name]
        if not score.get("n"):
            lines.append(f"| {title} | pending |")
        else:
            lines.append(
                f"| {title} | {score.get('r2_log1p_seconds', float('nan')):.4f} / {score.get('mae_seconds', float('nan')):.3f} s (n={score['n']}) |"
            )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
