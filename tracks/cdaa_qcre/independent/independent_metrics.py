"""Measure independently compiled/translated circuits with local duration maps.

The runtime column is an instruction-duration schedule estimate for IBM circuit
execution.  It is intentionally named ``estimated_execution_seconds`` rather
than a universal runtime target: no queue, workflow, or submit-to-result time is
included.
"""

from collections import defaultdict
from csv import DictWriter
import os
from pathlib import Path
import sys

from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parent
TRANSLATED = ROOT / "qasm" / "translated"
ORIGINAL = ROOT / "qasm" / "original"
UPSTREAM_CDAA = Path(
    os.environ.get("CDAA_UPSTREAM", str(ROOT.parents[1] / "cdaa"))
).resolve()
DURATIONS = Path(
    os.environ.get(
        "CDAA_DURATIONS",
        str(UPSTREAM_CDAA / "1_depth_runtime" / "data" / "instruction_durations"),
    )
).resolve()
OUT = ROOT / "metrics" / "independent_native_metrics.csv"

COMPILERS = {
    "sabre0330": ("eagle", "heron"),
    "sqgm": ("eagle", "heron"),
}

DEVICES = {
    "eagle": ("ibm_sherbrooke", "ibm_kyiv", "ibm_brisbane"),
    "heron": ("ibm_marrakesh", "ibm_kingston", "ibm_aachen"),
}

sys.path.insert(0, str(UPSTREAM_CDAA / "1_depth_runtime"))
from utils import translate_ibm_instruction_durations  # noqa: E402


def logical_widths():
    result = {}
    for path in ORIGINAL.glob("*.qasm"):
        result[path.stem] = QuantumCircuit.from_qasm_file(str(path)).num_qubits
    return result


def estimate_execution(circuit, duration_map):
    clocks = defaultdict(float)
    for inst, qubits, _ in circuit.data:
        name = inst.name
        if name in {"barrier", "measure", "reset"}:
            continue
        loc = tuple(circuit.find_bit(q).index for q in qubits)
        locations = duration_map[len(loc)]
        if loc in locations and name in locations[loc]:
            duration = locations[loc][name]
        elif len(loc) == 2 and loc[::-1] in locations and name in locations[loc[::-1]]:
            # Some snapshots record a calibrated two-qubit direction only;
            # use the reverse calibrated edge for the topology-proxy circuit.
            duration = locations[loc[::-1]][name]
        else:
            raise KeyError(f"No duration for {name}{loc}")
        start = max((clocks[q] for q in loc), default=0.0)
        finish = start + duration
        for q in loc:
            clocks[q] = finish
    return max(clocks.values(), default=0.0)


def main():
    widths = logical_widths()
    rows = []
    duration_cache = {}
    for compiler, architectures in COMPILERS.items():
        for architecture in architectures:
            devices = DEVICES[architecture]
            for device in devices:
                if device not in duration_cache:
                    path = DURATIONS / f"{device}_inst_dur.txt"
                    duration_cache[device] = translate_ibm_instruction_durations(str(path))
                duration_map = duration_cache[device]
                for circuit_name in sorted(widths):
                    path = TRANSLATED / compiler / architecture / device / f"{circuit_name}.qasm"
                    circuit = QuantumCircuit.from_qasm_file(str(path))
                    counts = circuit.count_ops()
                    rows.append(
                        {
                            "compiler": compiler,
                            "architecture": architecture,
                            "device_label": device,
                            "circuit": circuit_name,
                            "logical_width": widths[circuit_name],
                            "allocated_width": circuit.num_qubits,
                            "depth": circuit.depth(),
                            "total_ops": len(circuit.data),
                            "one_qubit_ops": sum(
                                value for name, value in counts.items() if name not in {"ecr", "cz"}
                            ),
                            "two_qubit_ops": counts.get("ecr", 0) + counts.get("cz", 0),
                            "estimated_execution_seconds": f"{estimate_execution(circuit, duration_map):.16g}",
                        }
                    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"rows={len(rows)}")
    for compiler, architectures in COMPILERS.items():
        for architecture in architectures:
            subset = [
                row for row in rows if row["compiler"] == compiler and row["architecture"] == architecture
            ]
            depths = [int(row["depth"]) for row in subset]
            runtimes = [float(row["estimated_execution_seconds"]) for row in subset]
            print(
                f"{compiler}/{architecture}: rows={len(subset)} "
                f"depth_min={min(depths)} depth_max={max(depths)} "
                f"execution_s_min={min(runtimes):.6g} execution_s_max={max(runtimes):.6g}"
            )
    print(f"output={OUT}")


if __name__ == "__main__":
    main()
