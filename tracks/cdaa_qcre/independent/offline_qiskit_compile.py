"""Run the Qiskit compiler path without IBM Runtime credentials.

FakeSherbrooke and FakeMarrakesh supply local target/coupling information for
the Eagle and Heron topology proxies.  The three device-labelled copies are
intentional: the historical artifact uses one topology route for all devices
in an architecture, while device-specific duration maps are applied later.
"""

from pathlib import Path
import shutil

from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dumps
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh, FakeSherbrooke


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "qasm" / "original"
DEST = ROOT / "qasm" / "compiled" / "qiskit141_proxy"

DEVICES = {
    "eagle": ("ibm_sherbrooke", "ibm_kyiv", "ibm_brisbane"),
    "heron": ("ibm_marrakesh", "ibm_kingston", "ibm_aachen"),
}


def main():
    backends = {"eagle": FakeSherbrooke(), "heron": FakeMarrakesh()}
    for architecture, devices in DEVICES.items():
        backend = backends[architecture]
        for source_path in sorted(SOURCE.glob("*.qasm")):
            circuit = QuantumCircuit.from_qasm_file(str(source_path))
            best = None
            best_depth = None
            for seed in range(5):
                candidate = transpile(
                    circuit,
                    backend=backend,
                    optimization_level=3,
                    seed_transpiler=seed,
                )
                depth = candidate.depth()
                if best is None or depth < best_depth:
                    best, best_depth = candidate, depth

            text = dumps(best)
            for device in devices:
                out = DEST / architecture / device / source_path.name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text, encoding="utf-8")
    print(f"compiled_files={len(list(DEST.glob('*/*/*.qasm')))}")


if __name__ == "__main__":
    main()
