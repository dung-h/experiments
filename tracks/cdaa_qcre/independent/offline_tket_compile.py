"""Run the TKET compiler path with the repository's local coupling graphs."""

from pathlib import Path
import os
import shutil
import sys

from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dumps

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "qasm" / "original"
TKET_SOURCE = ROOT / "qasm" / "tket_input"
DEST = ROOT / "qasm" / "compiled" / "tket_proxy"
DEVICES = {
    "heron": ("ibm_marrakesh", "ibm_kingston", "ibm_aachen"),
}

# comp_tket is already offline; it only uses the architecture graph in util.py.
# Resolve the sibling pinned cdaa checkout produced by bootstrap_upstreams.sh.
UPSTREAM_CDAA = Path(
    os.environ.get("CDAA_UPSTREAM", str(ROOT.parents[1] / "cdaa"))
).resolve()
sys.path.insert(0, str(UPSTREAM_CDAA / "0_compilation"))
from util import comp_tket  # noqa: E402


def main():
    # pytket 2.x treats the QASM ``cp`` instruction used by the benchmark
    # generators as CustomGate.  Decompose it with Qiskit into the standard
    # rz/sx/x/cx basis before invoking the repository's unchanged TKET pass
    # list; this preserves the unitary while avoiding a parser-version issue.
    TKET_SOURCE.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(SOURCE.glob("*.qasm")):
        circuit = QuantumCircuit.from_qasm_file(str(source_path))
        basic = transpile(
            circuit,
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=0,
            seed_transpiler=0,
        )
        (TKET_SOURCE / source_path.name).write_text(dumps(basic), encoding="utf-8")

    for architecture, devices in DEVICES.items():
        scratch = DEST / architecture / devices[0]
        scratch.mkdir(parents=True, exist_ok=True)
        comp_tket(architecture, "offline_proxy", 5, str(TKET_SOURCE), str(scratch))
        for device in devices[1:]:
            target = DEST / architecture / device
            target.mkdir(parents=True, exist_ok=True)
            for path in scratch.glob("*.qasm"):
                shutil.copy2(path, target / path.name)
    print(f"compiled_files={len(list(DEST.glob('*/*/*.qasm')))}")


if __name__ == "__main__":
    main()
