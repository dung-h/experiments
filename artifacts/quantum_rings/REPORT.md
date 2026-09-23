# Quantum Rings / iQuHACK 2026: contest, machine gap, and two public solutions

Finding recorded: 2026-09-18.
Packaged into this repository: 2026-09-23.

This note is the reporting path for the Quantum Rings track. It records the
contest task, a structural gap in the public labels, and an independent
artifact audit of Spirit Sprinters and SoftLocked. It does not rerun the
Quantum Rings SDK, does not use IBM QPU clocks, and does not merge these
rows with Ma–Li, Qonductor, CDAA/QCRE, or cuTensorNet.

Row-level outputs:

- [summary.json](summary.json)
- [community_spirit_sprinters_evaluation.json](community_spirit_sprinters_evaluation.json)
- [community_softlocked_evaluation.json](community_softlocked_evaluation.json)
- [community_softlocked_training_table_cv.json](community_softlocked_training_table_cv.json)

Clone instructions and pins: [tracks/quantum_rings/README.md](../../tracks/quantum_rings/README.md).

## 1. The contest

Event: iQuHACK 2026, Quantum Rings Circuit Fingerprint Challenge.
Official public repository: [iQuHACK/2026-Quantum-Rings](https://github.com/iQuHACK/2026-Quantum-Rings),
pinned at `1e2247f708136b2c6da99d7f98564ea2b15474f9`.
Related paper: [arXiv:2606.11620](https://arxiv.org/abs/2606.11620).
Official winner named by Quantum Rings: Spirit Sprinters
(Hayden Miller, Woody Hulse, Caden Schroeder, Rohan Pankaj, Patrick Jennings).

The challenge is offline. Teams do not run the simulator during scoring.
They submit code. Organizers run that code on hidden holdout QASM.

The public training table is `data/hackathon_public.json`: 36 OpenQASM 2.0
circuits and four execution contexts per circuit.

| Dimension | Values | Count |
|---|---|---:|
| Circuit | public QASM files, widths 7–130 | 36 |
| Backend tag | `CPU`, `GPU` | 2 |
| Precision tag | `single`, `double` | 2 |
| Labelled rows | circuit × backend × precision | **144** |

Each row asks for two predictions:

1. `predicted_threshold_min`: the smallest truncation-threshold ladder rung
   in `{1, 2, 4, 8, 16, 32, 64, 128, 256}` whose *mirror* fidelity is at
   least 0.99.
2. `predicted_forward_wall_s`: wall time of a 10,000-shot *forward* run at
   the selected threshold.

A mirror run is `U` followed by `U†`. The ideal output is the all-zero
bitstring. The sweep is comparatively cheap and is used only to choose the
threshold. The scored runtime is not that sweep. It is the expensive forward
simulation of `U`. Predicting below the true minimum rung scores zero on
that task; predicting above it is penalised by rung distance.

Two protocols exist in the public artifact and must not be pooled:

| Protocol | Threshold | Runtime label |
|---|---|---|
| Paper-style 0.75 | first mirror crossing of 0.75 | mirror-sweep wall time at that crossing (proxy) |
| Challenge/public 0.99 | first mirror crossing of 0.99 | 10,000-shot forward wall time at the public selected threshold |

This track scores community artifacts against the public 0.99 forward
target, and additionally reports Spirit Sprinters' threshold head against
both 0.75 and 0.99 labels because that repository's default training path
uses 0.75.

## 2. The public table does not describe the simulation machine

This is the contest's most important limitation for a runtime estimator.

The JSON schema for a result row is: circuit file, `backend`, `precision`,
status, threshold sweep, optional verify, optional state-setup, and the
forward run. Hardware-like fields that *are* present:

- `backend`: the string `CPU` or `GPU`;
- `precision`: the string `single` or `double`;
- `peak_rss_mb`: process resident set size for that run.

Hardware-like fields that are *absent* from every public row, from
`docs/DATA.md`, and from the submission interface:

- CPU model, core count, clock, AVX/AMX capability;
- GPU model, VRAM, driver, exclusive versus shared use;
- host DRAM, OS, filesystem;
- Quantum Rings SDK version that produced the labels;
- runner timeout policy, host name, NUMA layout.

`peak_rss_mb` on the public forward runs ranges from about 426 MB to
1,060 MB. That is a per-job memory observation, not a description of the
machine that collected the 144 times. CPU versus GPU is a categorical
feature, not a device profile.

The Quantum Rings SDK is separately documented with CPU, GPU and hybrid
backends, but installing today's package does not recover the hackathon
host. Any future credentialed rerun would be a new validation, with its
own SDK version and machine written down, not a reconstruction of the
contest labels.

Consequence for this project: these 144 times cannot be used to train a
hardware-aware simulator estimator of the kind we used for cuTensorNet on
the RTX 5070 Ti. They can be used to study *circuit-structure* predictors
that treat CPU/GPU and precision as tags. They cannot answer “how long
will this circuit take on this GPU”.

## 3. Spirit Sprinters

Repository: [woody-hulse/quantum-rings](https://github.com/woody-hulse/quantum-rings),
commit `14b5ea14dea8f6b55edf632030f5c022e8ad7373`.
Official winner. Public CLI and weights.

Two models, not one:

1. Tabular structural features from QASM (graph bandwidth, treewidth proxy,
   cut crossings, two-qubit span, long-range interaction ratio, light-cone
   spread, temporal entanglement velocity, gate n-grams) → XGBoost
   threshold classifier. The README still says CatBoost; the loaded
   artifact is `xgb_threshold.json`.
2. Qubit/gate interaction graph → edge-aware Transformer (4 layers, hidden
   width 32, 2 heads) trained on log₂ duration. The duration head receives
   the predicted threshold, the backend tag, and the precision tag.

The training default for the threshold head is fidelity 0.75. Duration
examples use the public `forward.run_wall_s`. The saved stack is therefore
not a clean 0.75-forward experiment.

We did not retrain. We loaded the committed artifacts and scored all 144
public rows on 2026-09-18.

Threshold accuracy:

| Label compared | Exact, known rows | Exact, all 144 rows including `star` | Within one ladder rung, known rows |
|---|---:|---:|---:|
| 0.75 | 99.3% (140 rows) | 96.5% | 99.3% |
| 0.99 | 86.1% (137 rows) | 81.9% | 97.1% |

Runtime against our public 0.99 forward target:

| Context | R² on log seconds | R² on raw seconds | Mean absolute error | Median relative error |
|---|---:|---:|---:|---:|
| All 144 rows | 0.743 | 0.561 | 107 s | 44% |
| GPU, single | 0.819 | −0.67 | 90 s | 27% |
| CPU, double | 0.733 | 0.750 | 100 s | 62% |
| GPU, double | 0.722 | 0.716 | 133 s | 68% |
| CPU, single | 0.608 | −3.20 | 104 s | 30% |

Log-second R² is the stable summary. Negative raw-second R² on two
contexts is caused by a few long circuits, not by a contradiction in the
log scores.

Concrete misses on the same 144-row table:

- `dj_indep_qiskit_130.qasm`, CPU, double: true 2,588 s, predicted 1,446 s.
- `qft_indep_qiskit_130.qasm`, GPU, double: true 2,060 s, predicted 1,008 s.
- `ae_indep_qiskit_130.qasm`, CPU, single: true 51 s, predicted 942 s;
  predicted threshold 8 against a 0.99 class that is not a normal rung.

Small circuits can be almost exact (`dj` 15-qubit CPU single: 5.83 s true
and predicted). The artifact is a strong public structural reference, not
a universal simulator estimator, and not a hidden-holdout score.

What is worth keeping from this solution: interaction-graph features as
proxies for tensor-network hardness; a separate threshold model; passing
CPU/GPU and precision into the duration model. The last point matters
because the contest itself never says *which* CPU or GPU.

## 4. SoftLocked

Repository: [SoftLocked/2026-Quantum-Rings](https://github.com/SoftLocked/2026-Quantum-Rings),
commit `da13c074d68c2fe3a1053c751234ee0916d73739`.
Also referred to in community notes as Team Popeye's.

Pipeline: about 90 tabular features from raw QASM → GradientBoosting
threshold classifier → GradientBoosting runtime regressor that takes the
predicted threshold. The CLI accepts precision and does **not** accept a
backend argument, so CPU and GPU rows with the same QASM and precision
receive the same prediction. Checkpoints
`models/threshold_classifier.pkl` and `models/runtime_model.pkl` are
present. `data/data.json` is missing, which blocks an exact retrain of the
original labels but does not block inference. Artifacts were serialised
with scikit-learn 1.8.0; the audit environment was 1.7.2.

In-domain control, rerun on the committed `training_data.csv` (1,107 rows,
576 circuits, CPU only) with the notebook's grouped split and the
published hyperparameters:

| Quantity | Value |
|---|---:|
| Threshold exact accuracy | 88.62% |
| Threshold competition score | 0.9066 |
| Runtime MAPE | 9.55% |
| Runtime R² on seconds | 0.9674 |
| Label range in that table | 3,193 s to 9.16×10⁶ s, median 74,048 s |

Those numbers match the repository README (MAPE ≈ 9.6%, R² ≈ 0.9666).
The checkpoint works in its own label domain.

The same checkpoint on our 144 public 0.99-forward rows:

| Scope | Threshold exact | R² on log seconds | Mean absolute error | Median prediction / median true |
|---|---:|---:|---:|---|
| All 144 rows | 10.4% | −25.5 | 2.60×10⁵ s | 92,530 s / 20 s |
| CPU 72 rows | 11.1% | −27.6 | 2.60×10⁵ s | same scale |

Example: `dj_indep_qiskit_130.qasm` GPU double is 2,474 s in the public
forward table and 2.29×10⁶ s under the SoftLocked checkpoint.

This is a domain mismatch, not a missing file. Some circuit filenames
overlap; the clocks do not. SoftLocked's table is thousands-to-millions
of seconds. The public forward labels are about 1–2,588 s, median 20 s.
The saved model cannot be compared with Spirit Sprinters on the 144-row
table until the runtime definition is aligned. The CLI's missing backend
argument is a second, independent defect relative to the contest schema.

What is worth keeping: grouped-by-file cross-validation, and the
demonstration that a tabular booster can look excellent in-domain while
being unusable on another clock. The architecture is not the next
estimator to copy.

## 5. How the two artifacts sit in this repository

| | Spirit Sprinters | SoftLocked |
|---|---|---|
| Role here | Public structural-model reference | In-domain tabular control and domain-shift warning |
| Consumes CPU/GPU tag | Yes | No |
| Public 144-row log-R² vs 0.99 forward | 0.743 | −25.5 |
| Own-table sanity check | Not rerun as CV; CLI ran 144/144 | R² 0.967 on `training_data.csv` |
| Machine profile used | None available; backend/precision tags only | None; precision only |

Bloch, Rishivarshil and Hazel were inspected in the 18 September 2026
audit. They are not drop-in predictors on this protocol and are not part
of this track. Their failure modes (0.75 vs 0.99, mirror vs forward,
row-level leakage, omitted scaler) are recorded only to keep them out of
the headline comparison.

## 6. Reproduction from this capsule

```bash
git clone https://github.com/dung-h/experiments.git
cd experiments
python3 scripts/verify_artifacts.py
bash tracks/quantum_rings/clone_upstreams.sh work
```

`verify_artifacts.py` checks that the committed report and JSON fixtures
are present and that the headline numbers in `summary.json` have not
drifted. It does not call the Quantum Rings SDK and does not require the
third-party clones.

Re-scoring the winner and SoftLocked artifacts requires those clones plus
the original evaluators that live beside the 18 September 2026 research
checkout. This capsule stores the independent outputs, not a copy of the
contest source tree and not a copy of the solution source trees.

## 7. Claims this track does not make

- It does not reconstruct the contest host.
- It does not treat Spirit Sprinters' 0.743 log-R² as a hidden-holdout
  result.
- It does not treat SoftLocked's 0.967 in-domain R² as evidence on the
  public 0.99-forward task.
- It does not compare either model with Ma–Li `result.time_taken` or with
  local RTX 5070 Ti cuTensorNet times.
