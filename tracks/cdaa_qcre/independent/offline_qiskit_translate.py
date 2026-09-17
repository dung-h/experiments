"""Translate Qiskit-proxy outputs with the same local fake targets."""

from pathlib import Path
import shutil

from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dumps
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh, FakeSherbrooke


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "qasm" / "compiled" / "qiskit141_proxy"
DEST = ROOT / "qasm" / "translated" / "qiskit141_proxy"
DEVICES = {
    "eagle": ("ibm_sherbrooke", "ibm_kyiv", "ibm_brisbane"),
    "heron": ("ibm_marrakesh", "ibm_kingston", "ibm_aachen"),
}


def main():
    backends = {"eagle": FakeSherbrooke(), "heron": FakeMarrakesh()}
    for architecture, devices in DEVICES.items():
        backend = backends[architecture]
        source_device = devices[0]
        for source_path in sorted((SOURCE / architecture / source_device).glob("*.qasm")):
            circuit = QuantumCircuit.from_qasm_file(str(source_path))
            translated = transpile(
                circuit,
                backend=backend,
                optimization_level=0,
                seed_transpiler=0,
            )
            text = dumps(translated)
            for device in devices:
                out = DEST / architecture / device / source_path.name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text, encoding="utf-8")
    print(f"translated_files={len(list(DEST.glob('*/*/*.qasm')))}")


if __name__ == "__main__":
    main()
