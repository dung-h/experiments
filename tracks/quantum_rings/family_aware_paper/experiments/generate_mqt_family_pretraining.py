"""Generate a public MQT Bench family-classifier pretraining set.

The paper describes approximately 200 independently synthesized MQT Bench
circuits, separate from the 36 evaluation circuits.  This script creates a
transparent public approximation: ten named MQT Bench algorithms at 20 qubit
sizes (5..24), with deterministic seeds and a manifest.  It must be run in an
environment containing ``mqt.bench``; the evaluation environment does not need
that package once the QASM files are generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import os

import numpy as np
from importlib.metadata import version
from mqt.bench import get_benchmark_alg
from qiskit import qasm2


_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ["QUANTUM_RINGS_ROOT"]).resolve() if os.environ.get("QUANTUM_RINGS_ROOT") else _SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "mqt_family_pretraining"

# These are the ten classifier families listed in Section III-D of the paper.
BENCHMARKS = {
    "DJ": "dj",
    "GHZ": "ghz",
    "W-State": "wstate",
    "Graph State": "graphstate",
    "Grover": "grover",
    "QFT": "qft",
    "QFT Entangled": "qftentangled",
    "QPE": "qpeexact",
    "QNN": "qnn",
    "VQE": "vqe_su2",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evaluation_hashes() -> set[str]:
    hashes = set()
    for path in (ROOT / "circuits").glob("*.qasm"):
        hashes.add(sha256_text(path.read_text(encoding="utf-8")))
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-qubits", type=int, default=5)
    parser.add_argument("--max-qubits", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output_dir = args.output_dir
    circuit_dir = output_dir / "circuits"
    circuit_dir.mkdir(parents=True, exist_ok=True)
    eval_hashes = evaluation_hashes()
    rows = []
    failures = []
    for family_index, (family, benchmark) in enumerate(BENCHMARKS.items()):
        for qubits in range(args.min_qubits, args.max_qubits + 1):
            sample_seed = args.seed + family_index * 1000 + qubits
            np.random.seed(sample_seed)
            try:
                circuit = get_benchmark_alg(
                    benchmark,
                    qubits,
                    random_parameters=True,
                )
                text = qasm2.dumps(circuit)
            except Exception as exc:  # keep the manifest explicit if a size is unsupported
                failures.append(
                    {
                        "family": family,
                        "benchmark": benchmark,
                        "n_qubits": qubits,
                        "seed": sample_seed,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            digest = sha256_text(text)
            duplicate_eval = digest in eval_hashes
            file_name = f"mqt_{benchmark}_n{qubits}_seed{sample_seed}.qasm"
            (circuit_dir / file_name).write_text(text, encoding="utf-8")
            rows.append(
                {
                    "file": file_name,
                    "family": family,
                    "benchmark": benchmark,
                    "n_qubits": int(circuit.num_qubits),
                    "depth": int(circuit.depth()),
                    "gate_count": int(len(circuit.data)),
                    "seed": sample_seed,
                    "level": "alg",
                    "qasm_sha256": digest,
                    "duplicate_eval_qasm": duplicate_eval,
                }
            )
    manifest = {
        "generated_at": datetime.now(timezone.utc).date().isoformat(),
        "generator": "experiments/generate_mqt_family_pretraining.py",
        "mqt_bench_version": version("mqt.bench"),
        "qiskit_version": version("qiskit"),
        "level": "alg",
        "random_parameters": True,
        "seed": args.seed,
        "families": list(BENCHMARKS),
        "requested_circuits": len(BENCHMARKS) * (args.max_qubits - args.min_qubits + 1),
        "generated_circuits": len(rows),
        "evaluation_qasm_hashes": len(eval_hashes),
        "failures": failures,
        "circuits": rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in (
        "generated_at", "mqt_bench_version", "qiskit_version",
        "requested_circuits", "generated_circuits", "failures",
    )}, indent=2))


if __name__ == "__main__":
    main()
