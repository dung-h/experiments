# CDAA/QCRE overlay and independent proxy

The two upstream repositories are cloned at the pins in upstream.lock.json.
The cdaa overlay replaces only the plotting fallback required when LaTeX is
absent. The independent directory contains the offline driver code written for
this study. It is not copied from the authors' repository.

Two environments are required:

| Purpose | Python/packages |
| --- | --- |
| Artifact analysis and offline Qiskit/TKET proxy | Python 3.10.21, Qiskit 1.4.1, qiskit-ibm-runtime 0.36.1, BQSKit 1.2.0, PyTKET 2.0.1, NumPy 1.26.4 |
| Original SQGM/SABRE driver | Python 3.10.21, qiskit-terra 0.46.3, NetworkX 3.4.2, NumPy 2.2.3 |

The bootstrap copies the 15 original QASM files into the independent work
directory. SQGM/SABRE use the historical artifact driver. Offline Qiskit/TKET
drivers use FakeSherbrooke/FakeMarrakesh topology proxies because the original
functions request a live IBM backend target.

The six `*_inst_dur.txt` files under `data/instruction_durations/` are the
archived snapshots used for the offline duration proxy. They are versioned here
because the upstream repository ignores that directory. They represent
instruction-duration inputs, not measured QPU wall-clock truth.
