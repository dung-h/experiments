#!/usr/bin/env python3
"""Build Ma--Li graph training data from the committed QASM and timing logs.

The upstream notebook assumes that a CSV contains the full QASM text in a
``quantum_circuit`` column.  The current repository stores circuit names and
keeps the QASM files separately, so this script performs the explicit join and
records an aligned manifest. By default it builds the offline Washington and
Sherbrooke track; Osaka/Kyoto can use a versioned properties cache (including
fake-backend snapshots) or opt-in IBM credentials.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from qiskit import QuantumCircuit

import circ_dag_converter
import helper


DEFAULT_DEVICES = ("washington", "sherbrooke")


def _circuit_name(row: pd.Series) -> str:
    for column in ("quantum_circuit", "circuit_name"):
        if column in row and pd.notna(row[column]):
            value = str(row[column])
            return Path(value).stem if value.endswith(".qasm") else value
    raise KeyError("Timing row has neither quantum_circuit nor circuit_name")


def _load_rows(data_dir: Path, devices: list[str], max_rows: int | None):
    frames: list[pd.DataFrame] = []
    for device in devices:
        path = data_dir / f"{device}_time_taken.csv"
        if not path.exists():
            raise FileNotFoundError(f"Timing log not found: {path}")
        frame = pd.read_csv(path)
        frame["source_device"] = device
        if max_rows is not None:
            frame = frame.head(max_rows)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _write_checkpoint(
    checkpoint: Path,
    samples: list,
    manifest_rows: list[dict],
    complete: bool,
) -> None:
    with checkpoint.open("wb") as handle:
        pickle.dump(
            {
                "complete": complete,
                "samples": samples,
                "manifest": pd.DataFrame(manifest_rows),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _build_one_sample(task):
    """Worker for one QASM-to-PyG conversion.

    Keeping this function at module scope makes it pickleable by
    ``ProcessPoolExecutor``.  The worker returns the manifest record together
    with the graph so the parent can checkpoint deterministically in input
    order.
    """
    data_dir, device, row_index, circuit_name, target, noise_dict = task
    data_dir = Path(data_dir)
    qasm_path = data_dir / "quantum_circuits" / f"{circuit_name}.qasm"
    if not qasm_path.exists():
        raise FileNotFoundError(f"QASM not found for {circuit_name}: {qasm_path}")
    qc = QuantumCircuit.from_qasm_file(str(qasm_path))
    global_features = helper.create_feature_dict(qc)
    graph = circ_dag_converter.circ_to_dag_with_data(
        qc,
        device,
        list(global_features.values()),
        n_qubit=127,
        noise_dict=noise_dict,
    )
    manifest = {
        "sample_index": row_index,
        "source_device": device,
        "circuit_name": circuit_name,
        "qasm_path": str(qasm_path.relative_to(data_dir.parent)),
        "target_column": "time_taken",
        "target_time_taken": float(target),
        "target_semantics": "Ma-Li upstream time_taken; device-labelled run",
    }
    return graph, manifest


def _build_device_dataset(
    data_dir: Path,
    device: str,
    max_rows: int | None,
    checkpoint: Path | None = None,
    checkpoint_every: int = 250,
    existing_samples: list | None = None,
    existing_manifest: pd.DataFrame | None = None,
    workers: int = 1,
    properties_dir: Path | None = None,
):
    rows = _load_rows(data_dir, [device], max_rows)
    noise_dict = circ_dag_converter.get_noise_dict(device, properties_dir)
    samples = list(existing_samples or [])
    manifest_rows = (
        existing_manifest.to_dict("records")
        if existing_manifest is not None
        else []
    )
    start_index = len(samples)
    pending = []
    for row_index, row in rows.iloc[start_index:].iterrows():
        device_name = str(row["source_device"]).lower()
        pending.append(
            (
                str(data_dir),
                device_name,
                int(row_index),
                _circuit_name(row),
                float(row["time_taken"]),
                noise_dict,
            )
        )

    def consume(results):
        for graph, manifest in results:
            samples.append(graph)
            manifest_rows.append(manifest)
            if checkpoint and len(samples) % checkpoint_every == 0:
                _write_checkpoint(checkpoint, samples, manifest_rows, complete=False)
                print(
                    f"saved partial checkpoint {checkpoint} ({len(samples)} samples)",
                    flush=True,
                )

    if workers > 1 and pending:
        print(f"building {len(pending)} {device} samples with {workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            consume(pool.map(_build_one_sample, pending, chunksize=1))
    else:
        consume(map(_build_one_sample, pending))
    print(f"built {len(samples)}/{len(rows)} samples", flush=True)
    return samples, pd.DataFrame(manifest_rows)


def _refine_and_standardize_in_place(samples: list):
    """Refine and standardize graphs without the upstream deep-copy spike.

    ``helper.refine_training_data`` and ``standardization_training_data`` each
    deep-copy the complete graph list and concatenate every node tensor.  The
    Ma--Li extracts contain enough graph nodes for those temporary copies to
    exceed the available RAM.  This equivalent streaming implementation keeps
    one graph list, computes sums/sums-of-squares, and then updates each graph
    in place.  The retained global columns follow the upstream positive-total
    rule; their means/stds are computed per feature (avoiding the upstream
    helper's accidental flattening of all graphs into one scalar stream).
    Standard deviations use the unbiased (N-1) estimator.
    """
    if not samples:
        return samples

    # Match refine_training_data: remove global-feature columns whose total is
    # not positive.  Accumulating only this small vector avoids a [n_graphs,
    # n_features] copy.
    first_gf = torch.as_tensor(samples[0].global_features, dtype=torch.float32)
    global_sum = torch.zeros_like(first_gf, dtype=torch.float64)
    for graph in samples:
        global_sum += torch.as_tensor(
            graph.global_features, dtype=torch.float32, device="cpu"
        ).to(dtype=torch.float64)
    non_zero = torch.nonzero(global_sum > 0, as_tuple=False).flatten().tolist()
    for graph in samples:
        gf = torch.as_tensor(graph.global_features, dtype=torch.float32)
        graph.global_features = gf[non_zero]

    # Streaming moments over all graph nodes, equivalent to cat(...).mean/std(0).
    first_x = torch.as_tensor(samples[0].x, dtype=torch.float32)
    x_sum = torch.zeros(first_x.shape[1], dtype=torch.float64)
    x_sumsq = torch.zeros_like(x_sum)
    x_count = 0
    gf_sum = torch.zeros(len(non_zero), dtype=torch.float64)
    gf_sumsq = torch.zeros_like(gf_sum)
    for graph in samples:
        x = torch.as_tensor(graph.x, dtype=torch.float32, device="cpu")
        x64 = x.to(dtype=torch.float64)
        x_sum += x64.sum(dim=0)
        x_sumsq += (x64 * x64).sum(dim=0)
        x_count += x.shape[0]
        gf = graph.global_features.to(dtype=torch.float64)
        gf_sum += gf
        gf_sumsq += gf * gf

    def moments(total, total_sq, count):
        mean = total / max(count, 1)
        if count > 1:
            variance = (total_sq - count * mean * mean) / (count - 1)
        else:
            variance = torch.zeros_like(mean)
        return mean.to(dtype=torch.float32), variance.clamp_min(0).sqrt().to(
            dtype=torch.float32
        )

    means_x, stds_x = moments(x_sum, x_sumsq, x_count)
    means_gf, stds_gf = moments(gf_sum, gf_sumsq, len(samples))
    for graph in samples:
        graph.x = (torch.as_tensor(graph.x, dtype=torch.float32) - means_x) / (
            1e-8 + stds_x
        )
        graph.global_features = (graph.global_features - means_gf) / (
            1e-8 + stds_gf
        )
    return samples


def build_dataset(
    data_dir: Path,
    devices: list[str],
    max_rows: int | None,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 250,
    workers: int = 1,
    properties_dir: Path | None = None,
):
    all_samples = []
    all_manifests = []
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for device in devices:
        checkpoint = checkpoint_dir / f"{device}_raw.pkl" if checkpoint_dir else None
        existing_samples = None
        existing_manifest = None
        checkpoint_complete = False
        if checkpoint and checkpoint.exists():
            print(f"loading checkpoint {checkpoint}", flush=True)
            with checkpoint.open("rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, dict):
                samples = payload["samples"]
                manifest = payload["manifest"]
                if not payload.get("complete", False):
                    existing_samples, existing_manifest = samples, manifest
                    samples, manifest = _build_device_dataset(
                        data_dir,
                        device,
                        max_rows,
                        checkpoint,
                        checkpoint_every,
                        existing_samples,
                        existing_manifest,
                        workers,
                        properties_dir,
                    )
                else:
                    checkpoint_complete = True
            else:
                # Backward-compatible tuple format from the first checkpoint version.
                samples, manifest = payload
                checkpoint_complete = True
        else:
            samples, manifest = _build_device_dataset(
                data_dir,
                device,
                max_rows,
                checkpoint,
                checkpoint_every,
                workers=workers,
                properties_dir=properties_dir,
            )
        if checkpoint and not checkpoint_complete:
            _write_checkpoint(checkpoint, samples, manifest.to_dict("records"), complete=True)
            print(f"saved checkpoint {checkpoint}", flush=True)
        all_samples.extend(samples)
        all_manifests.append(manifest)

    manifest_frame = pd.concat(all_manifests, ignore_index=True)
    standardized = _refine_and_standardize_in_place(all_samples)
    for graph, target in zip(standardized, manifest_frame["target_time_taken"].astype(float)):
        graph.y = torch.tensor(float(target), dtype=torch.float32)
    return standardized, manifest_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=list(DEFAULT_DEVICES),
        choices=["washington", "sherbrooke", "osaka", "kyoto"],
    )
    parser.add_argument("--max-rows-per-device", type=int, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional directory for per-device raw graph checkpoints.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel QASM/graph workers. Use a moderate value (e.g. 8); each worker imports Qiskit.",
    )
    parser.add_argument(
        "--backend-properties-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing <device>.json BackendProperties "
            "snapshots (for example fake Osaka/Kyoto)."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.data_dir / "training_data_standardization.npy"
    manifest = args.manifest or args.data_dir / "training_data_manifest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    samples, manifest_frame = build_dataset(
        args.data_dir,
        [d.lower() for d in args.devices],
        args.max_rows_per_device,
        args.checkpoint_dir,
        args.checkpoint_every,
        max(1, args.workers),
        args.backend_properties_dir,
    )
    with output.open("wb") as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest_frame.to_csv(manifest, index=False)

    print(
        f"saved {len(samples)} samples to {output} "
        f"(x={tuple(samples[0].x.shape)}, gf={tuple(samples[0].global_features.shape)})"
    )
    print(f"saved aligned manifest to {manifest}")


if __name__ == "__main__":
    # Make direct execution from the repository root resolve sibling modules.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
