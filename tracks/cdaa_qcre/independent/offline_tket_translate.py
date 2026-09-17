"""Translate TKET-proxy QASM to the local Heron fake target."""

from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dumps
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "qasm" / "compiled" / "tket_proxy" / "heron" / "ibm_marrakesh"
DEST = ROOT / "qasm" / "translated" / "tket_proxy" / "heron"
DEVICES = ("ibm_marrakesh", "ibm_kingston", "ibm_aachen")


def main():
    backend = FakeMarrakesh()
    for source_path in sorted(SOURCE.glob("*.qasm")):
        circuit = QuantumCircuit.from_qasm_file(str(source_path))
        translated = transpile(
            circuit,
            backend=backend,
            optimization_level=0,
            seed_transpiler=0,
        )
        text = dumps(translated)
        for device in DEVICES:
            out = DEST / device / source_path.name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
    print(f"translated_files={len(list((ROOT / 'qasm' / 'translated' / 'tket_proxy').glob('*/*/*.qasm')))}")


if __name__ == "__main__":
    main()
