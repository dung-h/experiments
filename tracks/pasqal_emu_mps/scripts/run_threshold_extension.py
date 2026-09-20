#!/usr/bin/env python3
"""Extend the local EMU-MPS threshold pilot with predeclared pulse families.

The artifact's target is *not* runtime.  For each pulse sequence it records
the smallest allowed ``max_bond_dim`` that reaches final-state fidelity 0.99
against a local high-chi reference.  The base 12-row sweep is intentionally
left untouched; this separate, resumable extension provides more known
families for an honest check of the family-residual pattern.
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
LAYOUTS = ((2, 4), (3, 4), (4, 4))

# These schedules and layouts are declared in source before execution. They
# extend, rather than replace, the four source-pilot families.
PULSE_FAMILIES = {
    "moderate": ((700, 3.0, 0.0),),
    "two_stage_ramp": ((300, 1.5, 0.0), (700, 4.0, 0.0)),
    "detuning_echo": ((500, 4.5, -4.0), (500, 4.5, 4.0)),
    "strong_relax": ((400, 5.0, 0.0), (600, 1.5, 0.0)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--target-fidelity", type=float, default=0.99)
    parser.add_argument("--dt-ns", type=float, default=10.0)
    parser.add_argument("--precision", type=float, default=1e-6)
    parser.add_argument("--max-krylov-dim", type=int, default=40)
    parser.add_argument("--reference-max-bond-dim", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def prior_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(json.loads(line)["case_id"])
        for line in path.read_text().splitlines()
        if line.strip()
    }


def sequence_for(family: str, rows: int, cols: int) -> tuple[Sequence, int, dict[str, float]]:
    pulses = PULSE_FAMILIES[family]
    register = Register.rectangle(rows, cols, spacing=5.0, prefix="q")
    sequence = Sequence(register, DigitalAnalogDevice)
    sequence.declare_channel("rydberg_global", "rydberg_global")
    for duration, amplitude, detuning in pulses:
        sequence.add(Pulse.ConstantPulse(duration, amplitude, detuning, 0.0), "rydberg_global")
    return sequence, sum(item[0] for item in pulses), {
        "pulse_duration_ns": float(sum(item[0] for item in pulses)),
        "pulse_segments": float(len(pulses)),
        "amplitude_max": float(max(item[1] for item in pulses)),
        "amplitude_mean": float(sum(item[1] for item in pulses) / len(pulses)),
        "detuning_max_abs": float(max(abs(item[2]) for item in pulses)),
        "detuning_mean": float(sum(item[2] for item in pulses) / len(pulses)),
    }


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


def write_csv(labels_path: Path, output_path: Path) -> None:
    labels = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
    if not labels:
        return
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this threshold extension requires a GPU.")
    if not 0.0 < args.target_fidelity <= 1.0:
        raise SystemExit("target fidelity must be in (0, 1]")
    warnings.filterwarnings("ignore", message="emu-mps allows only")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    labels_jsonl = args.output_prefix.with_name(args.output_prefix.name + "_labels.jsonl")
    sweep_jsonl = args.output_prefix.with_name(args.output_prefix.name + "_sweep.jsonl")
    completed = prior_case_ids(labels_jsonl) if args.resume else set()
    if not args.resume and (labels_jsonl.exists() or sweep_jsonl.exists()):
        raise SystemExit("Output exists; pass --resume to continue or choose a new prefix")

    definition = {
        "target_semantics": "smallest max_bond_dim rung reaching final-state fidelity against a local reference",
        "target_fidelity": args.target_fidelity,
        "reference_max_bond_dim": args.reference_max_bond_dim,
        "layouts": [list(layout) for layout in LAYOUTS],
        "families": {name: [list(pulse) for pulse in pulses] for name, pulses in PULSE_FAMILIES.items()},
        "rungs": list(RUNG_VALUES),
    }
    args.output_prefix.with_name(args.output_prefix.name + "_definition.json").write_text(json.dumps(definition, indent=2) + "\n")

    for family in PULSE_FAMILIES:
        for rows, cols in LAYOUTS:
            case_id = f"{family}__{rows}x{cols}"
            if case_id in completed:
                print(f"SKIP completed {case_id}", flush=True)
                continue
            sequence, duration_ns, features = sequence_for(family, rows, cols)
            n_qubits, end_us = rows * cols, duration_ns / 1000.0
            base = {
                "case_id": case_id,
                "family": family,
                "n_qubits": n_qubits,
                "rows": rows,
                "cols": cols,
                "dt_ns": args.dt_ns,
                "precision": args.precision,
                "target_fidelity": args.target_fidelity,
                **features,
            }
            print(f"REFERENCE {case_id} n={n_qubits}", flush=True)
            reference_start = time.perf_counter()
            reference = MPSBackend(
                sequence,
                config=config(args, args.reference_max_bond_dim, [StateResult(evaluation_times=[end_us])]),
            ).run()
            reference_wall = time.perf_counter() - reference_start
            reference_state = reference.get_result("state", end_us)
            reference_chi, reference_rss, reference_step_seconds = stats(reference)
            label: int | None = None
            for rung in RUNG_VALUES:
                if rung > args.reference_max_bond_dim:
                    continue
                started = time.perf_counter()
                result = MPSBackend(
                    sequence,
                    config=config(args, rung, [Fidelity(reference_state, evaluation_times=[end_us])]),
                ).run()
                wall_seconds = time.perf_counter() - started
                fidelity = float(result.get_tagged_results()["fidelity"][0])
                observed_chi, peak_rss_mb, step_seconds = stats(result)
                reached = fidelity >= args.target_fidelity
                append_jsonl(sweep_jsonl, {
                    **base, "rung": rung, "fidelity": fidelity, "reached_target": reached,
                    "observed_max_chi": observed_chi, "peak_rss_mb": peak_rss_mb,
                    "step_seconds": step_seconds, "wall_seconds": wall_seconds,
                    "reference_max_chi": reference_chi, "reference_wall_seconds": reference_wall,
                })
                print(f"  rung={rung} fidelity={fidelity:.6f} chi={observed_chi} wall={wall_seconds:.2f}s", flush=True)
                if reached:
                    label = rung
                    break
            append_jsonl(labels_jsonl, {
                **base,
                "required_max_bond_dim": label if label is not None else "*",
                "reference_max_chi": reference_chi,
                "reference_peak_rss_mb": reference_rss,
                "reference_step_seconds": reference_step_seconds,
                "reference_wall_seconds": reference_wall,
                "reference_censored": reference_chi >= args.reference_max_bond_dim,
            })
            write_csv(labels_jsonl, args.output_prefix.with_name(args.output_prefix.name + "_labels.csv"))
    environment = {
        "python": __import__("platform").python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "target_semantics": definition["target_semantics"],
    }
    args.output_prefix.with_name(args.output_prefix.name + "_environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(f"WROTE {labels_jsonl}")


if __name__ == "__main__":
    main()
