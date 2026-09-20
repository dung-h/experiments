# Simulator papers: provenance and local data-generation protocol

**Logged:** 20 September 2026
**Purpose:** turn public simulator artifacts into controlled local runtime labels. This is a supplement to, not a replacement for, the QPU/runtime-service tracks in this repository.

## The three papers under review

| ID | Paper | Public source state | What can be reproduced locally without external credentials |
| --- | --- | --- | --- |
| SP-1 | [VQCSim: a GPU-oriented variational-circuit simulator](https://arxiv.org/abs/2607.11985) | [`Security-FIT/VQCSim`](https://github.com/Security-FIT/VQCSim), pinned at `7fc274d448c5e67a7a4149088afa594203cd5707` | The generic Torch lowering, Qiskit parity checks, and RQ2 GPU time/VRAM suite. |
| SP-2 | [Transpilation-Aware Runtime Prediction for Noisy Quantum Circuit Simulation](https://arxiv.org/abs/2609.12980) | Paper public; author code and measured 1,402-circuit runtime table were not linked at the time of this check | An independent reconstruction using the Ma--Li circuit corpus, current Qiskit fake backends, and noisy Aer. This is the existing `azizov_independent` track. |
| SP-3 | [Benchmarking Zero-Setup Quantum Circuit Simulators](https://arxiv.org/abs/2607.09882) | [`arulrhikm/mps-pps-zero-setup-benchmarks`](https://github.com/arulrhikm/mps-pps-zero-setup-benchmarks), pinned at `c4485f1c267410d291a717d05aa12d493559f337` | The artifact's local Qiskit `pauli-prop` comparison. The source's MPS/statevector benchmark runners submit to the BlueQubit service, so their reported cloud runtimes cannot be recreated locally or mixed with local labels. |

The two upstream checkouts are intentionally ignored under `work/`; the exact URLs and revisions are recorded in [`upstream/upstream.lock.json`](../../upstream/upstream.lock.json). No upstream source code is redistributed here.

## Why generate simulator labels locally

Simulator data are not intrinsically scarce. The constraint is that a runtime label is meaningful only under one declared execution contract. Every generated row must retain at least:

- simulator/framework and version;
- CPU or GPU, GPU model, thread settings, precision, and memory policy;
- circuit source/family/hash and logical/compiled features;
- what timing includes (preparation, first call, warm call, sampling, or a particular kernel);
- warm-up, repeat count, timeout and terminal status.

Rows from Aer noisy execution, Torch/VQCSim statevector inference, tensor-network contraction and MPS/Pauli propagation are separate targets. They may share circuit descriptors, but they must not be concatenated into a generic `runtime_seconds` column.

## SP-1: VQCSim local GPU label set

The VQCSim RQ2 runner measures repeated **inference** of a prepared Torch-native circuit. Its timed stage excludes construction/lowering (`prepare_ms` remains a separate field) and uses CUDA synchronization around the repeated calls. It is therefore a local GPU inference-kernel target, not an Aer target or full application wall-clock.

The initial local matrix uses the five variational families from the source artifact (`qaoa`, `qnn`, `vqe_real_amp`, `vqe_su2`, `vqe_two_local`), batch size 1 and float32 on an RTX 5070 Ti. It has Qiskit parity enabled. The raw checkpointed output is:

```text
artifacts/simulator_papers/vqcsim_rq2_gpu_fp32_16_24_20260920/
```

Recreate it after bootstrapping the pinned source and creating the isolated Python 3.13 environment documented below:

```bash
ROOT=/path/to/quantum-runtime-estimator-replications
SRC="$ROOT/work/vqcsim"
PYTHONPATH="$SRC/code" "$ROOT/work/venvs/vqcsim/bin/python" -u "$SRC/code/run_mqtbench_rq2_suite.py" \
  --output_dir "$ROOT/artifacts/simulator_papers/vqcsim_rq2_gpu_fp32_16_24_20260920" \
  --allow_relaxed_defaults \
  --families qaoa,qnn,vqe_real_amp,vqe_su2,vqe_two_local \
  --frameworks vqcsim --batch_sizes 1 --modes inference \
  --no_discover_sizes --size_min 16 --size_max 24 \
  --precision_mode float32 --iters 3 --warmup 1 --repeats 3 \
  --per_job_timeout_seconds 300 --checkpoint_every_n 1 --skip_plots
```

The deliberately small first batch is a **data-generation canary**, not a performance comparison against the paper: this workstation has one RTX 5070 Ti, while the paper reports multiple frameworks and five GPU types. A later matched experiment must vary batch size and add a separately declared CPU/precision matrix; no number here may be used to rank frameworks whose timing contracts differ.

## SP-2: Azizov independent reconstruction

The current independent matrix already supplies a usable generated simulator table rather than merely a paper review:

- target: noisy Qiskit Aer execution after fake-backend transpilation; transpilation and simulator setup are recorded separately;
- source: 149 locally screened circuits, two fake-backend snapshots, four optimization levels;
- completed data: 1,192 rows, all terminal status `ok` in the q<=9 screened matrix;
- primary split: logical-circuit-group split, not row-random split;
- best hybrid Random Forest: log-R2 `0.8558` grouped-circuit, `0.8375` family-held-out; backend transfer is lower (`0.5511` and `0.6297`), which is the more realistic warning signal.

Exact commands, raw rows and split reports are in [`tracks/azizov_independent`](../../tracks/azizov_independent) and [`artifacts/azizov_independent`](../../artifacts/azizov_independent). The missing author artifact means these numbers must be described as an independent reproduction, never as the paper's reproduced score.

## SP-3: Zero-Setup local Pauli-propagation canary

The upstream artifact contains a genuine local runner, `pauli_path_tests/experiments/pps_benchmark_qiskit.py`. It constructs a 127-qubit heavy-hex transverse-field-Ising circuit, converts it once to `pauli-prop` rotation gates, then times only `propagate_through_rotation_gates`. The bounded runner in this capsule uses the same circuit, observable, gate order, conversion and timer while allowing a safe explicit delta/trial subset:

```bash
ROOT=/path/to/quantum-runtime-estimator-replications
"$ROOT/work/venvs/vqcsim/bin/python" -m pip install pauli-prop
"$ROOT/work/venvs/vqcsim/bin/python" \
  "$ROOT/experiments/simulator_papers/run_zero_setup_pps_canary.py" \
  --upstream-root "$ROOT/work/zero_setup" \
  --output "$ROOT/artifacts/simulator_papers/zero_setup_pps_local_canary_20260920.jsonl" \
  --deltas 1e-2 --trials 5
```

`propagation_seconds` is the target. `circuit_build_seconds` and `conversion_seconds` are provenance fields and must remain outside it. This local Pauli-propagation label is neither a statevector runtime nor the BlueQubit remote MPS runtime reported elsewhere in SP-3.

## Supplemental independent probes (SP-4 through SP-6)

Three additional papers were checked after the initial three tracks. Their
author artifacts do not permit an exact numerical replication here, so the
repository records bounded local probes with raw data and explicit limits:

- [`SIMULATOR_ESTIMATOR_VALIDATION_REPORT_2026-09-20.md`](../../artifacts/simulator_papers/SIMULATOR_ESTIMATOR_VALIDATION_REPORT_2026-09-20.md) is the current formal result table, source map and failure analysis.
- [`DEEP_ANALYSIS_2026-09-20.md`](../../artifacts/simulator_papers/DEEP_ANALYSIS_2026-09-20.md) records residual, calibration-shift, feasibility and numerical-regime diagnostics; explanations are explicitly non-causal.
- `explicit_plan_ranking_expanded_20260920/` contains 98 feasible candidate-plan timings and seven separately recorded infeasible candidates. It is an analytical minimum-FLOP baseline, not a learned ranker.
- `emu_mps_threshold_extension_20260920/` plus `family_aware_emu_mps_expanded_20260920/` contain 24 nested known-family threshold labels/evaluation. The target is a fidelity threshold, not runtime.
- `precision_selection_expanded_20260920/` contains 52 complex64/complex128 pairs and an out-of-fold safety-policy check.

The report supplies exact commands. These remain separate simulator/threshold
targets and are not added to the VQCSim, Aer or Pauli-propagation label sets.

## Local environment

The initial SP-1/SP-3 environment is deliberately separate from Aer/CUDA-Q environments:

```bash
ROOT=/path/to/quantum-runtime-estimator-replications
mkdir -p "$ROOT/work/venvs"
/path/to/python3.13 -m venv "$ROOT/work/venvs/vqcsim"
"$ROOT/work/venvs/vqcsim/bin/python" -m pip install --upgrade pip
"$ROOT/work/venvs/vqcsim/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
"$ROOT/work/venvs/vqcsim/bin/python" -m pip install mqt.bench==2.1.0 qiskit==2.5.2 nvidia-ml-py
```

Record the resolved package versions from each run's `manifest.json`; do not assume a future Torch/Qiskit version will produce identical wall-clock values.

## Immediate next increments

1. SP-1 now has a 90-row base/mirror evaluation. Add independent circuit seeds, batch sizes and a separate CPU/precision matrix before relying on its apparently strong small-corpus scores.
2. SP-3 now has a 36-row controlled propagation grid. Expand `delta` only with a declared time/RSS guard; the current delta-held-out failure says that interpolation and extrapolation must be reported separately.
3. SP-2 needs backend/domain adaptation or a shifted calibration method: nominal 90% intervals cover only 21.7% in fake-backend-held-out evaluation.
4. The plan and precision probes need independent candidate/hardware diversity before a learned ranker or automatic selector is attempted. The EMU-MPS track needs a non-constructed corpus before it can assess unseen-family transfer.
5. Build a separate matched CPU/GPU x float32/float64 statevector matrix. It will be a new dataset with a new target contract, not a pooled extension of these records.
