#!/usr/bin/env python3
"""Generate EMU-MPS threshold labels for a small QCRE-style pilot.

For each pulse sequence, a high-bond-dimension reference state is generated
first.  The simulator is then run at increasing bond-dimension rungs and the
smallest rung reaching the requested final-state fidelity is recorded.  This
is a threshold label, not the unconstrained bond dimension trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import torch
from emu_mps import Fidelity, MPSBackend, MPSConfig, StateResult
from pulser import Pulse, Register, Sequence
from pulser.devices import DigitalAnalogDevice


RUNG_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/pasqal_emu_mps/threshold_sweep"),
    )
    parser.add_argument("--target-fidelity", type=float, default=0.99)
    parser.add_argument("--dt-ns", type=float, default=10.0)
    parser.add_argument("--precision", type=float, default=1e-6)
    parser.add_argument("--max-krylov-dim", type=int, default=40)
    parser.add_argument("--reference-max-bond-dim", type=int, default=256)
    return parser.parse_args()


def sequence_for(family: str, rows: int, cols: int) -> tuple[Sequence, int, dict[str, float]]:
    register = Register.rectangle(rows, cols, spacing=5.0, prefix="q")
    sequence = Sequence(register, DigitalAnalogDevice)
    sequence.declare_channel("rydberg_global", "rydberg_global")
    if family == "weak":
        pulses = [(400, 1.5, 0.0)]
    elif family == "strong":
        pulses = [(1000, 5.0, 0.0)]
    elif family == "detuned":
        pulses = [(1000, 5.0, -5.0)]
    elif family == "two_stage":
        pulses = [(500, 5.0, 0.0), (500, 2.0, 5.0)]
    else:
        raise ValueError(f"Unknown family: {family}")
    for duration, amplitude, detuning in pulses:
        sequence.add(
            Pulse.ConstantPulse(duration, amplitude, detuning, 0.0),
            "rydberg_global",
        )
    features = {
        "pulse_duration_ns": float(sum(item[0] for item in pulses)),
        "pulse_segments": float(len(pulses)),
        "amplitude_max": float(max(item[1] for item in pulses)),
        "amplitude_mean": float(sum(item[1] for item in pulses) / len(pulses)),
        "detuning_max_abs": float(max(abs(item[2]) for item in pulses)),
        "detuning_mean": float(sum(item[2] for item in pulses) / len(pulses)),
    }
    return sequence, int(features["pulse_duration_ns"]), features


def config(args: argparse.Namespace, max_bond_dim: int, observables):
    return MPSConfig(
        dt=args.dt_ns,
        precision=args.precision,
        max_bond_dim=max_bond_dim,
        max_krylov_dim=args.max_krylov_dim,
        num_gpus_to_use=1,
        optimize_qubit_ordering=False,
        observables=observables,
        log_level=40,
    )


def stats(result) -> tuple[int, float, float]:
    values = result.get_tagged_results()["statistics"]
    return (
        max(int(item["max_bond_dimension"]) for item in values),
        max(float(item["RSS"]) for item in values),
        sum(float(item["duration"]) for item in values),
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this sweep requires a GPU.")
    warnings.filterwarnings("ignore", message="emu-mps allows only")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    cases = [
        (family, size, size if size != 12 else 4)
        for family in ("weak", "strong", "detuned", "two_stage")
        for size in (8, 12, 16)
    ]
    # Use compact rectangular layouts: 8=2x4, 12=3x4, 16=4x4.
    cases = []
    for family in ("weak", "strong", "detuned", "two_stage"):
        cases.extend(
            [
                (family, 2, 4),
                (family, 3, 4),
                (family, 4, 4),
            ]
        )

    threshold_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for family, rows, cols in cases:
        sequence, duration_ns, features = sequence_for(family, rows, cols)
        n_qubits = rows * cols
        end_us = duration_ns / 1000.0
        base = {
            "family": family,
            "n_qubits": n_qubits,
            "rows": rows,
            "cols": cols,
            "dt_ns": args.dt_ns,
            "precision": args.precision,
            "target_fidelity": args.target_fidelity,
            **features,
        }
        print(f"REFERENCE family={family} n={n_qubits}", flush=True)
        reference_start = time.perf_counter()
        reference = MPSBackend(
            sequence,
            config=config(
                args,
                args.reference_max_bond_dim,
                [StateResult(evaluation_times=[end_us])],
            ),
        ).run()
        reference_wall = time.perf_counter() - reference_start
        reference_state = reference.get_result("state", end_us)
        reference_chi, reference_rss, reference_step_seconds = stats(reference)
        label = None
        for rung in RUNG_VALUES:
            if rung > args.reference_max_bond_dim:
                continue
            start = time.perf_counter()
            result = MPSBackend(
                sequence,
                config=config(args, rung, [Fidelity(reference_state, evaluation_times=[end_us])]),
            ).run()
            wall = time.perf_counter() - start
            fidelity = float(result.get_tagged_results()["fidelity"][0])
            observed_chi, peak_rss, step_seconds = stats(result)
            reached = fidelity >= args.target_fidelity
            threshold_rows.append(
                {
                    **base,
                    "rung": rung,
                    "fidelity": fidelity,
                    "reached_target": reached,
                    "observed_max_chi": observed_chi,
                    "peak_rss_mb": peak_rss,
                    "step_seconds": step_seconds,
                    "wall_seconds": wall,
                    "reference_max_chi": reference_chi,
                    "reference_wall_seconds": reference_wall,
                }
            )
            print(
                f"  rung={rung} fidelity={fidelity:.6f} observed_chi={observed_chi} "
                f"wall={wall:.2f}s",
                flush=True,
            )
            if reached and label is None:
                label = rung
                break
        label_rows.append(
            {
                **base,
                "required_max_bond_dim": label if label is not None else "*",
                "reference_max_chi": reference_chi,
                "reference_peak_rss_mb": reference_rss,
                "reference_step_seconds": reference_step_seconds,
                "reference_wall_seconds": reference_wall,
                "reference_censored": reference_chi >= args.reference_max_bond_dim,
            }
        )

    threshold_path = args.output_prefix.with_suffix(".csv")
    labels_path = args.output_prefix.with_name(args.output_prefix.name + "_labels").with_suffix(".csv")
    with threshold_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)
    with labels_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(label_rows[0]))
        writer.writeheader()
        writer.writerows(label_rows)
    environment = {
        "python": __import__("platform").python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "emu_mps": "2.9.1",
        "pulser_core": "1.9.1",
        "threshold_rungs": list(RUNG_VALUES),
        "reference_max_bond_dim": args.reference_max_bond_dim,
        "target_fidelity": args.target_fidelity,
        "label_semantics": "smallest max_bond_dim rung reaching final-state fidelity against reference",
    }
    args.output_prefix.with_suffix(".environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )
    print(f"WROTE {threshold_path}")
    print(f"WROTE {labels_path}")


if __name__ == "__main__":
    main()
