"""Translate independent SQGM/SABRE QASM with local Qiskit fake backends.

This is an offline supplement to the paper artifact.  It deliberately uses a
single fake backend per IBM topology (FakeSherbrooke for Eagle and
FakeMarrakesh for Heron), because the original translation script asks the IBM
Runtime API for historical device names.  The resulting files are therefore
labelled as topology proxies, not as exact historical backend compilations.
"""

from pathlib import Path
import shutil

from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dumps
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh, FakeSherbrooke


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "qasm" / "compiled"
DST = ROOT / "qasm" / "translated"

DEVICES = {
    "eagle": ("ibm_sherbrooke", "ibm_kyiv", "ibm_brisbane"),
    "heron": ("ibm_marrakesh", "ibm_kingston", "ibm_aachen"),
}


def translate_tree():
    fake_by_arch = {
        "eagle": FakeSherbrooke(),
        "heron": FakeMarrakesh(),
    }

    # The independent SQGM/SABRE run emits one topology-level circuit and the
    # upstream sorter copies it to three device-labelled directories.  We
    # translate each unique topology/compiler/circuit once, then preserve that
    # layout for downstream scripts.
    for compiler in ("sabre0330", "sqgm"):
        for architecture in ("eagle", "heron"):
            backend = fake_by_arch[architecture]
            source_dir = SRC / compiler / architecture
            source_devices = [source_dir / name for name in DEVICES[architecture]]
            for source_device in source_devices:
                for source_path in sorted(source_device.glob("*.qasm")):
                    # Only translate once for each circuit in a topology.  The
                    # output is copied to all historical device labels below.
                    rel_name = source_path.name
                    translated = None
                    for device_name in DEVICES[architecture]:
                        device = source_dir / device_name
                        target = DST / compiler / architecture / device.name / rel_name
                        if target.exists():
                            translated = target
                            break
                    if translated is None:
                        circuit = QuantumCircuit.from_qasm_file(str(source_path))
                        compiled = transpile(
                            circuit,
                            backend=backend,
                            optimization_level=0,
                            seed_transpiler=0,
                        )
                        text = dumps(compiled)
                        translated = DST / compiler / architecture / source_device.name / rel_name
                        translated.parent.mkdir(parents=True, exist_ok=True)
                        translated.write_text(text, encoding="utf-8")

                    for device_name in DEVICES[architecture]:
                        device = source_dir / device_name
                        target = DST / compiler / architecture / device.name / rel_name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target != translated:
                            shutil.copy2(translated, target)


if __name__ == "__main__":
    translate_tree()
    files = list(DST.glob("*/*/*/*.qasm"))
    print(f"translated_files={len(files)}")
