# Replication report: Tremba et al. (QCE 2025)

**Paper:** Matthew Tremba, Paul Hovland, and Ji Liu, *Is Circuit Depth
Accurate for Comparing Quantum Circuit Runtimes?* (QCE 2025 / [arXiv
2505.16908](https://arxiv.org/abs/2505.16908)).

**Repositories:**

- `qcre`: standalone quantum circuit runtime estimator
- `cdaa`: software and data artifact for the paper

**Status:** the bundled-data analysis has been reproduced locally, and all
compiler paths have now been exercised offline from the 15 raw QASM circuits.
SQGM/SABRE use the repository's original Terra-0.46.3 driver; Qiskit and TKET
use credential-free topology proxies described below.

## Scope and provenance

The artifact contains:

- 15 original benchmark circuits (VQE, QAOA, Hamiltonian simulation, and QFT),
  ranging from 4 to 64 qubits;
- compiled and translated QASM for IBM Eagle and Heron devices;
- depth/runtime CSVs for six devices: Sherbrooke, Kyiv, Brisbane, Marrakesh,
  Kingston, and Aachen;
- twelve instruction-duration snapshots from IBM backends;
- the authors' figures and textual statistics under `data/2_analysis/out/`.

The upstream paper describes four compiler paths: SABRE, SQGM, TKET, and
Qiskit. Eagle analysis contains three compilers because the TKET translation
was not usable for those devices; Heron analysis contains all four. The paper
uses 90 pairwise comparisons per Heron device and 45 per Eagle device.

## Environment

The reproducibility environment is `/home/server/Documents/.venv-cdaa-new`,
installed from `new_requirements.txt` and `qcre/requirements.txt`:

| Package | Version |
|---|---:|
| Python | 3.10.21 |
| Qiskit | 1.4.1 |
| qiskit-ibm-runtime | 0.36.1 |
| BQSKit | 1.2.0 |
| PyTKET | 2.0.1 |
| NumPy | 1.26.4 |
| Matplotlib | 3.10.1 |

The original SQGM/SABRE compilation stage additionally specifies Qiskit Terra
0.46.3, NumPy 2.2.3, and NetworkX 3.4.2. This environment is installed at
`/home/server/Documents/.venv-cdaa-old` and was used for the independent run.

## Reproduction performed

The supplied data were copied into the working directories expected by the
scripts. Stable instruction-duration aliases were created because
`1_depth_runtime/data.py` expects names such as
`ibm_sherbrooke_inst_dur.txt`, while the artifact stores timestamped files.

The following steps completed successfully:

```sh
cd /home/server/Documents/cdaa/2_analysis
/home/server/Documents/.venv-cdaa-new/bin/python runstats.py > out/stats.txt
MPLBACKEND=Agg /home/server/Documents/.venv-cdaa-new/bin/python \
  runfigs.py
```

The figure script originally required an external `latex` executable. It now
uses Matplotlib mathtext when LaTeX is unavailable; plots otherwise retain the
upstream styling and filenames.

The standalone QCRE estimator was also exercised directly on the supplied
translated `qft4` circuit for IBM Sherbrooke. Its estimate was
`1.0737777777777776e-05` seconds, matching the corresponding published
`verify.csv` value.

### Independent raw-QASM compilation

The original 15 QASM files were copied to
`/home/server/Documents/cdaa_independent/0_compilation/qasm/original` and
compiled with the repository's unmodified SQGM/SABRE driver:

```sh
cd /home/server/Documents/cdaa_independent/0_compilation
PATH=/home/server/Documents/.venv-cdaa-old/bin:$PATH python -u compile.py
```

The run completed for both topology configurations and produced 180 QASM
files: 15 circuits × 2 compilers (`sabre0330`, `sqgm`) × 2 architectures
(`eagle`, `heron`) × 3 device-labelled output directories. The three device
directories within an architecture contain the same topology-level output,
as in the upstream SQGM artifact; they are labels for the subsequent device
analysis, not separate routing runs.

The independent output was parsed with Qiskit Terra 0.46.3. Width is the
allocated backend width (127 qubits for Eagle and 156 for Heron), while the
logical benchmark widths remain 4, 8, 16, 32, and 64 qubits. Across the 45
circuits per compiler/architecture, observed depth ranges were:

| Compiler/topology | Output depth range | Output 2-qubit-gate range |
|---|---:|---:|
| SABRE / Eagle | 8–2793 | 3–13752 |
| SQGM / Eagle | 8–1765 | 3–16086 |
| SABRE / Heron | 8–3354 | 3–14130 |
| SQGM / Heron | 8–1867 | 3–16428 |

The largest independent case is QFT-64 (depth 2793/1765 on Eagle and
3354/1867 on Heron for SABRE/SQGM). These values are compiler outputs, not
runtime measurements; runtime estimation still requires the matching
instruction-duration snapshot and must preserve the paper's circuit-execution
target semantics.

For an offline end-to-end smoke of the next stage, the independently compiled
QASM was translated with Qiskit 1.4.1 using `FakeSherbrooke` (Eagle proxy) and
`FakeMarrakesh` (Heron proxy), then scored with the archived local IBM
instruction-duration snapshots. This produced another 180 files under
`/home/server/Documents/cdaa_independent/0_compilation/qasm/translated/` and
180 rows in
`/home/server/Documents/cdaa_independent/0_compilation/metrics/independent_native_metrics.csv`.
Across the 45 rows in each compiler/topology group, Pearson correlation of
translated depth with the duration-map execution estimate was 0.9963 (SABRE,
Eagle), 0.9967 (SQGM, Eagle), 0.9995 (SABRE, Heron), and 0.9975 (SQGM, Heron).
These correlations are a diagnostic for this generated proxy set, not a
claim about universal runtime prediction or real QPU observations.

### Independent Qiskit/TKET proxy runs

The public artifact is sufficient to avoid IBM credentials for analysis: its
instruction-duration snapshots and translated QASM are already local, and
`1_depth_runtime/data.py` is configured for local mode. The original Qiskit
compiler/translation functions nevertheless call `QiskitRuntimeService` to
fetch a live backend target, so separate offline drivers were added rather
than modifying the paper's source code. They use `FakeSherbrooke` as an Eagle
target and `FakeMarrakesh` as a Heron target, with fixed transpiler seeds and
the same five-repetition/lowest-depth selection rule.

```sh
cd /home/server/Documents/cdaa_independent/0_compilation
/home/server/Documents/.venv-cdaa-new/bin/python offline_qiskit_compile.py
/home/server/Documents/.venv-cdaa-new/bin/python offline_qiskit_translate.py
/home/server/Documents/.venv-cdaa-new/bin/python offline_tket_compile.py
/home/server/Documents/.venv-cdaa-new/bin/python offline_tket_translate.py
/home/server/Documents/.venv-cdaa-new/bin/python independent_all_metrics.py
```

This produced 90 Qiskit-proxy compiled files, 90 Qiskit-proxy translated
files, 45 TKET-proxy compiled files, and 45 TKET-proxy translated files. The
combined metrics file contains 315 rows across SABRE, SQGM, Qiskit-proxy, and
TKET-proxy. Qiskit-proxy had the shortest translated depth for all 15 circuits
on each device label in this local experiment; this is expected from its
optimization-level-3 objective and is not evidence that it wins on the
historical IBM devices.

## Results

### Runtime estimator verification

The supplied verification data contain 15 circuits × 3 Eagle compiler paths:

- 30 of 45 estimates match the Qiskit pulse-schedule duration exactly;
- the largest absolute difference is `3.4998046127832083e-16` seconds;
- the corresponding runtime is `0.00183830044444444` seconds.

This is floating-point noise and reproduces the paper's conclusion that the
QCRE estimator is equivalent to the scheduler for this verification set.

### Relative runtime-change prediction

The statistic is `%RE = |Δdepth - Δruntime| / |Δruntime| × 100`, evaluated over
all compiler pairs for each base circuit. Median `%RE` from the supplied data:

| Device | Traditional | Multi-qubit | Gate-aware (average weight) | Best manual weight |
|---|---:|---:|---:|---:|
| IBM Sherbrooke | 28.199% | 8.151% | 2.254% | 0.913% (`w=0.11`) |
| IBM Kyiv | 42.902% | 6.966% | 1.087% | 0.566% (`w=0.09`) |
| IBM Brisbane | 31.630% | 7.916% | 0.323% | 0.215% (`w=0.09`) |
| IBM Marrakesh | 42.950% | 14.949% | 0.920% | 0.100% (`w=0.53`) |
| IBM Kingston | 46.957% | 14.157% | 0.264% | 0.026% (`w=0.47`) |
| IBM Aachen | 65.919% | 19.922% | 8.554% | 7.484% (`w=0.47`) |

The mean gate-aware error reduction calculated by the artifact scripts is:

- `63.74×` relative to traditional depth (`64×` rounded);
- `17.80×` relative to multi-qubit depth (`18×` rounded).

The current arXiv abstract says `68×` and `18×`, while its detailed results
section says `64×` and `18×`. The packaged `stats.txt` and this rerun agree
with the detailed result (`63.74×`, rounding to `64×`). This discrepancy is
recorded rather than silently reconciled.

### Identifying the shortest-runtime compiler version

Using the average architecture weight map, gate-aware depth identifies the
runtime-optimal compiled version on:

- Sherbrooke: 100%;
- Kyiv: 100%;
- Brisbane: 100%;
- Marrakesh: 100%;
- Kingston: 100%;
- Aachen: 80%.

Thus it is perfect on five of six devices. The average improvement is 20
percentage points over traditional depth and 43.33 points over multi-qubit
depth. For Marrakesh, 8 of the 10 incorrect multi-qubit identifications are
caused by ties in multi-qubit depth.

### Architecture weights

The reproduced average weight maps are:

| Architecture | ECR | CZ | RZ | SX/X |
|---|---:|---:|---:|---:|
| IBM Eagle | 1.000 | n/a | 0.000 | 0.0942 |
| IBM Heron | n/a | 1.000 | 0.000 | 0.483 |

## Interpretation and limits

This paper studies a depth-like metric for comparing **compiled circuit
versions**. Its runtime target is circuit execution duration estimated from
device instruction durations; it is not queue time, workflow turnaround, or a
pooled cross-source machine-learning target.

The present replication is strong for the published analysis because the
authors' compiled/translated QASM, instruction-duration snapshots, CSVs, and
reference plots are included. All four compiler paths now have credential-free
offline runs, but only SQGM/SABRE use the paper's exact historical topology
driver. Qiskit/TKET outputs use fake-backend/topology proxies and therefore
must not be presented as exact recompilations of the six historical IBM
backends. Any future report must state whether it uses archived QASM/CSV,
exact raw-QASM output, or a proxy output, and must not treat device-labelled
copies as distinct routing experiments.

## Canonical outputs

- Numerical output: `2_analysis/out/stats.txt`
- Reproduced figures: `2_analysis/out/*.png`
- Archived reference figures/statistics: `data/2_analysis/out/`
- Standalone estimator: `qcre/estimate.py`
- Independent SQGM/SABRE outputs: `/home/server/Documents/cdaa_independent/0_compilation/qasm/compiled/`
- Independent offline compiler/translation scripts: `/home/server/Documents/cdaa_independent/0_compilation/offline_qiskit_compile.py`, `/home/server/Documents/cdaa_independent/0_compilation/offline_qiskit_translate.py`, `/home/server/Documents/cdaa_independent/0_compilation/offline_tket_compile.py`, `/home/server/Documents/cdaa_independent/0_compilation/offline_tket_translate.py`
- Independent offline translation/metrics: `/home/server/Documents/cdaa_independent/0_compilation/offline_translate.py`, `/home/server/Documents/cdaa_independent/0_compilation/independent_metrics.py`, `/home/server/Documents/cdaa_independent/0_compilation/independent_all_metrics.py`
- Combined proxy metrics: `/home/server/Documents/cdaa_independent/0_compilation/metrics/independent_all_metrics.csv`
- Full artifact instructions: `readme.md`

Future additions should append a dated section and record the repository
commit, environment, input directory, compiler set, backend, instruction
duration source, and whether results came from archived or newly generated
QASM. Do not overwrite the archived results.
