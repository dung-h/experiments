# Finding clusters

Packaged: 2026-09-23.

This capsule recorded many related runs. They are easier to read as three
clusters than as a flat list of folders. Scores from different clusters are
not a leaderboard. The clocks are different.

| Cluster | Question | Label | Headline |
| --- | --- | --- | --- |
| 1. Ma–Li × Azizov | After the circuit is transpiled to a backend, can we predict recorded Osaka/Kyoto `result.time_taken`? | Ma–Li hardware `result.time_taken` (1024 shots); Washington/Sherbrooke simulator `time_taken` stays a separate clock | Compiled structure is the transferable object. Logical depth is not. Simulator pretraining plus an affine head does not beat a model trained on the 340 hardware rows. QWalk remains the failure tail. |
| 2. Quantum Rings solutions | Do the public Spirit Sprinters and SoftLocked artifacts predict the contest's 144-row 0.99-forward table? | Public 10,000-shot forward wall time at the 0.99 threshold | Winner log-R² 0.743 on the public table. SoftLocked is in-domain on its own table (R² 0.967) and a domain-shift failure here (log-R² −25.5). The contest does not name the simulation machine. |
| 3. Family-aware paper | Does the public iQuHACK artifact reproduce [arXiv:2606.11620](https://arxiv.org/abs/2606.11620)? | Paper-style 0.75 mirror-sweep *proxy*, and public 0.99 forward; never pooled | Trees are the strongest public runtime baseline (0.75 log-R² 0.757; 0.99 ExtraTrees 0.712). The reconstructed family MLP is not a competitive runtime model on this artifact. Paper headline R² 0.82 is not recovered. |

Older capsules (Qonductor, CDAA/QCRE, cuTensorNet, the independent Azizov Aer screen, dense-statevector, CUDA-Q) remain in this repository. They are not part of the three-cluster reading path. Track-level tables stay in [`RESULTS.md`](../RESULTS.md).

Do not pool Ma–Li `result.time_taken`, contest 0.75 mirror time, contest 0.99 forward time, SoftLocked's 74k-second table, or local RTX 5070 Ti cuTensorNet times.

---

## Cluster 1 · Ma–Li × Azizov

**Finding dates:** 2026-09-17 through 2026-09-22.

This is one scientific object: *compiled / transpiled QPU runtime estimation*
on the Ma–Li Osaka/Kyoto labels. Azizov is not a second paper-track with its
own published table. Azizov contributes the representation change: replace the
logical DAG with the circuit after `transpile(..., target backend)`.

Ma–Li's published Graph Transformer still encodes the *source* DAG and then
adapts onto hardware. Azizov argues that noisy-simulator wall-clock after
transpilation is a compiled-circuit problem. The runs below keep Ma–Li's
labels and two-stage protocol, and swap in compiled inputs.

### What the two papers actually do

Ma–Li pretrains a Graph Transformer on FakeWashington / FakeSherbrooke
simulator `time_taken`, then fine-tunes on 340 recorded Osaka/Kyoto
`result.time_taken` rows. The graph is built from the logical QASM. Backend
properties (T1/T2 and related calibration fields) are attached as node
features. The paper's qualitative gap is pretrained ≈ 0.59 versus
from-scratch ≈ 0.89 on hardware labels.

Azizov studies Qiskit Aer noisy-simulation time *after* transpilation. The
authors' code and `T_exec` table are not public. The independent Aer screen
already in this capsule (`artifacts/azizov_independent/`) is a local
feasibility matrix, not Azizov's HPC numbers. Cluster 1 does not replace that
screen. It asks a different question: if Ma–Li's transfer protocol is rerun
with transpiled circuits, does compiled structure close the gap?

Current FakeOsaka / FakeKyoto / FakeWashingtonV2 / FakeSherbrooke snapshots
are **not** the historical job-day calibrations. Every compiled score in this
cluster is a current-snapshot proxy.

### How to read the folders

| Folder | Role |
| --- | --- |
| [`artifacts/mali/`](../artifacts/mali/) | Original seed-1234 fold-10 QWalk failure of the *logical* pretrained model |
| [`artifacts/validation/mali_qcre_proxy/`](../artifacts/validation/mali_qcre_proxy/) | 340-row FakeOsaka/Kyoto compiled proxy versus observed labels |
| [`artifacts/validation/mali_direct_estimator/`](../artifacts/validation/mali_direct_estimator/) | Direct Ridge on compiled scalars, trained on hardware only |
| [`artifacts/validation/mali_deep_protocol_v1/`](../artifacts/validation/mali_deep_protocol_v1/) | Leakage-safe split audit (170 QASM hashes, 300 physical graphs) |
| [`artifacts/validation/mali_physical_dag_v1/`](../artifacts/validation/mali_physical_dag_v1/) | Native DAG corpus, static ablation, QWalk forensics, topology controls |
| [`artifacts/validation/mali_family_ood_v1/`](../artifacts/validation/mali_family_ood_v1/) | Experiment A / B2 / C: missing families versus unseen hardware families |
| [`artifacts/validation/mali_azizov_compiled_transfer_v1/`](../artifacts/validation/mali_azizov_compiled_transfer_v1/) | Transpile → Ridge on compiled scalars → affine transfer onto QPU |
| [`artifacts/validation/mali_azizov_transpiled_dag_v1/`](../artifacts/validation/mali_azizov_transpiled_dag_v1/) | Same protocol on coarsened native DAGs (ignore `qpu_scratch_pilot/`) |
| [`artifacts/validation/mali_strict_transfer_logical_v1/`](../artifacts/validation/mali_strict_transfer_logical_v1/) | Remove all 170 hardware QASM hashes from simulator pretraining |
| [`artifacts/mali/graph_intervention_audit_20260921/`](../artifacts/mali/graph_intervention_audit_20260921/) | Graph-feature intervention audit |

The native-DAG coarsened tensors under
`mali_azizov_transpiled_dag_v1/coarsened/` are rebuildable and are not
committed.

### Headline numbers

Logical pretrained Ma–Li, seed 1234, fold 10: R² **−1.90** on the held-out
Osaka QWalk row `qwalk-noancilla_indep_qiskit_9`. From-scratch on the same
split: R² **0.94**. That is negative transfer, not a generic GNN failure.

Compiled proxy on the same 340 hardware labels (current FakeOsaka/Kyoto,
optimization level 1, seed 1234):

| Input | Typical log-R² on hardware labels |
| --- | ---: |
| Logical depth | about −0.02 to negative |
| Physical depth / compiled T1 | about 0.79–0.86 |
| Compiled T1 + weighted path (T2) | about 0.87 |
| Rich static native-DAG summary | 0.890 grouped / 0.876 strict / 0.870 family-component |
| QPU-scratch coarsened native DAG | **0.905** (n=340) |

Simulator Washington/Sherbrooke `time_taken` is a different clock (median
0.83 s, max 73 s). Compiled Ridge T2 reaches log-R² **0.435** in-domain.
The coarsened native DAG reaches **0.779**. That jump is larger on the
simulator clock than on the QPU clock, because the hardware labels already
sit near a compiled-size ceiling.

Transfer from simulator compiled features onto the 340 QPU rows, affine head,
same-circuit protocol:

| Model | log-R² / MAE |
| --- | --- |
| QPU scratch compiled T2 | **0.875 / 0.527 s** |
| Sim compiled T2 → affine QPU (`incl`) | 0.353 / 1.290 s |
| Sim compiled T2 → affine QPU (`excl`) | 0.349 / 1.286 s |
| Sim native DAG → affine QPU | **−2.63 / 17.8 s** (two QWalk rows explode) |
| Sim native DAG → fine-tune QPU | 0.875 / 0.557 s |

`incl` keeps the test QASM in simulator pretraining. `excl` removes it. The
two differ by about 0.004 log-R². Seeing the same logical file on
Washington/Sherbrooke does not help Osaka/Kyoto once compiled features are
already on both sides. Fine-tune from the simulator DAG almost matches QPU
scratch (0.875 versus 0.905). That is initialization, not frozen transfer.

Affine DAG transfer fails because of two rows of
`qwalk-noancilla_indep_qiskit_9`: predictions 3,196 s (Osaka) and 2,497 s
(Kyoto) against labels ≈ 13.5 s. Median absolute error of that affine map is
still 0.97 s. R² is destroyed by the tail. Dropping those two rows raises
affine log-R² only to 0.37. QWalk is outside the support of the training
families: Experiment A places it nearer the 12 *missing* MQT families than
the 10 hardware-selected ones.

### Family-OOD (A / B2 / C)

Experiment A does not need new QPU jobs. The 12 MQT families absent from the
hardware table occupy a different region: median two-qubit count 81 versus
4,606; interaction density 0.034 versus 1.0. Compiled FakeOsaka/Kyoto widens
the split. Historical Experiment B (score those 12 families on retired
Osaka/Kyoto) remains blocked: there are no labels.

Experiment C holds out each of the 9 hardware family components from *both*
simulator pretraining and QPU training. Excluding QWalk, scratch compiled T1
reaches log-R² **0.859**. Logical T0 is **−1.283**. Simulator logical T0 plus
affine QPU is **−0.282**. Compiled representation, not family identity, is
what generalises inside the selected region.

Experiment B2 puts the held family back into simulator pretraining and still
holds it out of QPU training. Compiled-proxy T2 `incl` is 0.248 versus `excl`
0.225, still far behind scratch T1 0.859. Seeing the family on the simulator
does not replace hardware labels.

### What this cluster can and cannot claim

Supported:

- Logical depth is not a QPU runtime estimator on this table.
- After transpile, physical depth, two-qubit count, duration/criticality and
  native DAG size carry the signal.
- Scratch training on the 340 hardware rows currently beats simulator
  pretraining plus affine calibration.
- QWalk is an out-of-support tail tied to calibration/compiled-size mismatch.
- Duplicate QASM/backend labels already differ by a median 0.42 s, so a
  point estimate should be reported with an interval and an OOD flag.

Not supported:

- That a GAT / TransformerConv / true DAG GNN beats the static compiled
  model. Topology controls currently favour a node-only coarsened baseline.
- That FakeOsaka/Kyoto recover historical job-day calibration.
- That the 12 missing MQT families have been scored on real QPU.
- That Azizov's published Aer `T_exec` numbers were reproduced.
- A universal QPU runtime estimator.

### Replication

Ma–Li QASM and labels come from the pinned
`Quantum-Execution-Time-Prediction` checkout (`scripts/bootstrap_upstreams.sh`).
Cluster scripts live under `experiments/`:

```bash
python experiments/mali_qcre_proxy.py --mali-root work/mali
python experiments/run_mali_direct_estimator.py
python experiments/mali_split_audit.py
python experiments/evaluate_mali_azizov_compiled_transfer.py
python experiments/evaluate_mali_azizov_transpiled_dag_transfer.py
python experiments/mali_family_feature_space_audit.py
python experiments/evaluate_mali_unseen_family_transfer.py
python experiments/evaluate_mali_simulator_assisted_family_transfer.py
```

Exact flags and environments are in each folder's `REPORT.md` / `PROTOCOL.md`.
`python3 scripts/verify_artifacts.py` checks the committed reports and
headline JSON; it does not retrain.

---

## Cluster 2 · Quantum Rings contest solutions

**Finding recorded:** 2026-09-18. **Packaged:** 2026-09-23.

Canonical report: [`artifacts/quantum_rings/REPORT.md`](../artifacts/quantum_rings/REPORT.md).

Event: iQuHACK 2026, Quantum Rings Circuit Fingerprint Challenge.
Official tree: [iQuHACK/2026-Quantum-Rings](https://github.com/iQuHACK/2026-Quantum-Rings)
at `1e2247f708136b2c6da99d7f98564ea2b15474f9`. Official winner named by
Quantum Rings: Spirit Sprinters.

The public table is 36 OpenQASM circuits × `{CPU, GPU}` × `{single, double}`
= **144 rows**. Each row asks for (1) the minimum truncation-threshold ladder
rung whose *mirror* fidelity is at least 0.99, and (2) the 10,000-shot
*forward* wall time at that rung. A mirror run is `U` then `U†`. The scored
runtime is not the mirror sweep.

### The contest does not describe the simulation machine

Present: `backend` as the string CPU or GPU, `precision` as single or
double, and per-run `peak_rss_mb`. Absent from every public row and from
`docs/DATA.md`: CPU model, core count, GPU model, VRAM, host DRAM, OS, the
Quantum Rings SDK version that produced the labels, and the runner timeout.
CPU versus GPU is a categorical tag, not a device profile. These 144 times
cannot train a hardware-conditioned simulator estimator of the kind used for
cuTensorNet on the RTX 5070 Ti.

The Quantum Rings SDK was not rerun. Holdout QASM is private. Installing
today's PyPI package would be a new validation, not a reconstruction of the
contest host.

### Spirit Sprinters

Repository [woody-hulse/quantum-rings](https://github.com/woody-hulse/quantum-rings)
at `14b5ea14dea8f6b55edf632030f5c022e8ad7373`. Two models: tabular structural
features → XGBoost threshold classifier, then a qubit/gate interaction-graph
Transformer on log₂ duration. The duration head receives the predicted
threshold and the CPU/GPU and precision tags.

Public 144-row 0.99-forward result: log-R² **0.743**, MAE **107 s**, median
relative error 44%. Threshold exact accuracy is 99.3% against 0.75 labels and
86.1% against known 0.99 rows. Large-circuit misses remain (`dj` 130-qubit
CPU double: 2,588 s true, 1,446 s predicted). This is a public-artifact
cross-check, not a hidden-holdout score.

Worth keeping: interaction-graph features as tensor-network hardness proxies;
a separate threshold model; passing CPU/GPU and precision into the duration
head even though the contest never names the chips.

### SoftLocked

Repository [SoftLocked/2026-Quantum-Rings](https://github.com/SoftLocked/2026-Quantum-Rings)
at `da13c074d68c2fe3a1053c751234ee0916d73739`. About 90 tabular features →
GradientBoosting threshold and runtime models. The CLI accepts precision and
does **not** accept a backend argument.

On the author `training_data.csv` (1,107 rows, CPU only, median 74,048 s)
grouped CV matches the README: threshold exact 88.6%, seconds R² **0.967**.
On the public 144-row 0.99-forward table the same checkpoint has log-R²
**−25.5** and MAE 2.60×10⁵ s (median prediction 92,530 s versus public median
20 s). That is a clock mismatch, not a missing file. SoftLocked is a
domain-shift warning, not the next architecture to copy.

Bloch, Rishivarshil and Hazel were inspected on 18 September 2026 and are
outside this cluster.

### Replication

```bash
bash tracks/quantum_rings/clone_upstreams.sh work
python3 scripts/verify_artifacts.py
```

`verify_artifacts.py` checks the committed JSON fixtures. Re-scoring the
third-party checkpoints needs those clones plus the 18 September evaluators;
this capsule stores the independent outputs, not a copy of the solution trees.

---

## Cluster 3 · Family-aware paper (arXiv:2606.11620)

**Finding recorded:** 2026-09-18.

The notes existed in the local research checkout of `2026-Quantum-Rings`.
They are packaged here under
[`artifacts/quantum_rings/family_aware_paper/`](../artifacts/quantum_rings/family_aware_paper/).
The paper is related to the same contest as Cluster 2, but it is a different
claim: a family-conditioned residual/FiLM network for threshold selection and
runtime on Quantum Rings labels. Cluster 2 audits winner checkpoints.
Cluster 3 reconstructs the paper protocol on the public 144 rows.

Start with
[`REPLICATION_SUMMARY_REPORT.md`](../artifacts/quantum_rings/family_aware_paper/REPLICATION_SUMMARY_REPORT.md).

### Two targets that must not be pooled

| Protocol | Threshold | Runtime label |
| --- | --- | --- |
| Paper-style 0.75 | first mirror crossing of 0.75 | mirror-sweep wall time at that crossing (**proxy**) |
| Challenge/public 0.99 | first mirror crossing of 0.99 | 10,000-shot forward wall time at the public selected threshold |

The public JSON was generated at 0.99. A true paper-protocol 10,000-shot
forward label at 0.75 is not present. Four 0.75 rows use a documented
fallback. CPU/GPU ordering reverses between the two targets (0.75 GPU-single
log-R² 0.888 versus 0.99 GPU-double 0.284 on Random Forest). Pooling them
invents a third clock.

### What we reran

Circuit-level 5-fold CV, seed 0: all four CPU/GPU × precision rows of one
QASM stay in the same fold. Features: 32 paper-shaped statistics, including
interaction-graph and RCM/cut-pressure approximations. Models: Random Forest,
ExtraTrees, Gradient Boosting, no-family MLP, family-conditioned MLP, and a
frozen public MQT family classifier.

Paper-style 0.75 (mirror-sweep proxy):

| Model | Threshold exact | Runtime log-R² |
| --- | ---: | ---: |
| Random Forest | 68.1% | 0.748 |
| Gradient Boosting runtime | — | **0.757** |
| Family MLP, predicted family | 59.0% | **−0.253** |

Challenge 0.99 (public forward):

| Model | Threshold exact | Runtime log-R² |
| --- | ---: | ---: |
| Random Forest | 61.1% | 0.417 |
| ExtraTrees runtime | — | **0.712** |
| Family MLP, predicted family | 63.2% | **−0.856** |

Fold-local family classifier accuracy on the 36 evaluation circuits is
**39%**. A frozen MLP trained on an independent public 200-circuit MQT set
reaches **39/40 (97.5%)** on its own MQT holdout — the same *number* the
paper reports, not the authors' private set. Transfer onto the 36 contest
circuits is 33% overall and 46% among mapped known families. Frozen MQT
conditioning does not beat trees on Quantum Rings runtime.

The paper reports 79.5% exact threshold, 91.2% within one rung, and runtime
R² 0.82. Those values are not independently reproduced. Blockers: 0.75 versus
0.99 targets; missing 0.75 forward labels; unpublished training code/weights;
unpublished exact family-pretraining circuits; no hidden-holdout truth.

### Replication

Public JSON/QASM: clone the official challenge tree at the Cluster 2 pin.

```bash
bash tracks/quantum_rings/clone_upstreams.sh work
export QUANTUM_RINGS_ROOT="$PWD/work/quantum_rings_challenge"
mkdir -p "$QUANTUM_RINGS_ROOT/results"
python tracks/quantum_rings/family_aware_paper/experiments/replicate_paper.py \
  --target both --seed 0 \
  --output "$QUANTUM_RINGS_ROOT/results/paper_replication.json"
python tracks/quantum_rings/family_aware_paper/experiments/protocol_audit.py \
  --target both --seed 0 \
  --output "$QUANTUM_RINGS_ROOT/results/protocol_audit.json"
```

Environment: `tracks/quantum_rings/family_aware_paper/experiments/requirements-reproduction.txt`.
The 200 MQT QASM files are not committed here; `generate_mqt_family_pretraining.py`
can rebuild them. The frozen classifier used for the packaged integration
result is
`artifacts/quantum_rings/family_aware_paper/mqt_family_classifier/`.
Compare reruns to the JSON files in that folder, not to the paper PDF.

The official iQuHACK remote is read-only. Do not push research commits there.

---

## Other capsules, not these three clusters

Already published in this repository, with their own reports:

- Qonductor public regression versus numerical DAG, plus mapping audit
- CDAA/QCRE schedule-duration agreement
- cuTensorNet `RUNTIME_EST` versus RTX 5070 Ti CUDA-event clocks, including the 90% QFT width frontier and the Ma–Li MQT QASM q≤10 table (`artifacts/cutensornet/mali_qasm_qle10_v1_20260923/`)
- Azizov independent noisy-Aer feasibility matrix (`q≤9` local screen)
- Dense-statevector PyTorch kernel (v1/v2)
- CUDA-Q dense and MPS pilots

The 90% QFT frontier and the Ma–Li q≤10 QASM contraction table are part of
the cuTensorNet capsule, not a fourth finding cluster. Uncommitted local
follow-ups (CPU-only dense-statevector eval, cuTensorNet multi-target
calibration, paired CUDA-Q/cuTensorNet IR, runtime-estimate trace) stay out
of this packaging round on purpose.

Day-by-day record: [`DAILY_WORK_LOG_2026-09-17_TO_2026-09-22.md`](DAILY_WORK_LOG_2026-09-17_TO_2026-09-22.md).
Removed credential-gated plans: [`REMOVED_AND_OUT_OF_SCOPE.md`](REMOVED_AND_OUT_OF_SCOPE.md).
