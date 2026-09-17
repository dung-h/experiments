# Qonductor-SC25: independent reproduction report

This is the detailed report preserved by the four-track replication capsule.
Its script/output paths refer to the patched Qonductor checkout, not to this
directory. Use [`REPRODUCTION.md`](../../REPRODUCTION.md) to materialize that
checkout at the pinned revision before running it.

Date: 2026-09-12
Repository: [`manosgior/Qonductor-SC25`](https://github.com/manosgior/Qonductor-SC25)
Checked-out commit: `5d1ac8a` (`main`)
Paper: [Qonductor: A Hybrid Quantum-Classical Cloud Orchestrator (arXiv:2408.04312)](https://arxiv.org/abs/2408.04312), with the SC'25 manuscript available as [Qonductor-SC-2025.pdf](https://dse.in.tum.de/wp-content/uploads/2025/09/Qonductor-SC-2025.pdf).

## What was reproduced

This is a public-artifact reproduction. The repository already contains the
CSV/JSON/SQLite artifacts produced by the paper's experiments, so the headline
plots and aggregate values can be recomputed without submitting new jobs to
IBM Quantum. The reproduction covers the resource estimator, end-to-end
scheduler comparison, IBM status analysis, scheduling-manager analysis, the
SQLite dataset distributions, and the regression feature plot.

The offline scheduler smoke test also ran one two-circuit QAOA-style job on
two current Qiskit Runtime fake backends (`fake_algiers` and `fake_cairo`). It
returned one assignment and scheduler metadata successfully.

For a small local smoke test, set `QONDUCTOR_TRANSPILE_PROCESSES=1` and
`QISKIT_NUM_PROCS=1`; the normal parallel path remains the default for larger
benchmark sweeps.

The exact smoke command is also captured in
`experiments/fake_scheduler_smoke.py` in that patched checkout, which writes
`results/fake_scheduler_smoke.json` there.

## Environment and commands

The run used Python 3.10.21 in `/home/server/Documents/.venv-qonductor` with
current packages including Qiskit 2.5.2, Qiskit IBM Runtime 0.49.0,
MQT Bench 2.2.3, NumPy 2.2.6, pandas 2.3.3, and scikit-learn 1.7.2.
`mqt.predictor` was deliberately omitted: it is not imported by the public
artifact analyses and its dependency resolution pulls a very large Torch/CUDA
stack. No IBM token or network API call is required for the results below.

From the repository root:

```bash
PYTHON=/home/server/Documents/.venv-qonductor/bin/python
PYTHONPATH=. "$PYTHON" \
  experiments/reproduce_public_results.py

MPLBACKEND=Agg PYTHONPATH=. "$PYTHON" \
  src/analysis/estimator_analysis.py
MPLBACKEND=Agg PYTHONPATH=. "$PYTHON" \
  src/analysis/e2e_performance.py
MPLBACKEND=Agg PYTHONPATH=. "$PYTHON" \
  src/analysis/ibm_status_analysis.py
MPLBACKEND=Agg PYTHONPATH=. "$PYTHON" \
  src/analysis/scheduling_manager_analysis.py
MPLBACKEND=Agg PYTHONPATH=. "$PYTHON" \
  src/analysis/dataset_analysis.py
MPLBACKEND=Agg PYTHONPATH=. "$PYTHON" \
  src/analysis/feature_analysis.py
```

`experiments/reproduce_public_results.py` writes the machine-readable summary
to `results/reproduction_metrics.json` in that checkout. Reproduced plots are
kept separately under `plots/reproduced/` so the checked-in paper figures are
not overwritten. The pinned package subset used for this offline path is
listed in `experiments/requirements-reproduction.txt`.

## Recomputed values

The values below are from the committed CSV artifacts. Execution-time units
follow the repository figure label (milliseconds); the Pareto runtime and
end-to-end JCT columns are seconds.

### Resource estimator

| Task | n | MAE | RMSE | R² | MAPE | Bias (prediction − target) |
|---|---:|---:|---:|---:|---:|---:|
| Qonductor execution estimate vs real | 100 | 502.359 ms | 1059.620 ms | 0.9386 | 7.659% | −20.412 ms |
| DAG/numerical estimate vs real | 100 | 915.609 ms | 1451.502 ms | 0.8849 | 14.410% | +144.069 ms |
| Qonductor fidelity estimate vs real | 100 | 0.07148 | 0.10292 | 0.7932 | 21.326% | −0.00338 |
| Numerical/model fidelity vs real | 100 | 0.11539 | 0.15576 | 0.5263 | 30.756% | +0.01848 |

The fidelity/runtime table contains 25 points, with mean runtime 103.016 s and
mean fidelity 0.4845; the reproduced Pareto mask marks 5 nondominated points.

### End-to-end scheduling artifact

The 100-row public traces give:

- Qonductor mean JCT: **5,675 s**;
- FCFS mean JCT: **10,925 s**;
- mean JCT reduction: **48.055%**;
- Qonductor mean fidelity: **0.72958** vs FCFS **0.74970**, a **2.684% relative loss**;
- mean utilization: **74.583%** vs **44.957%**, a gain of **29.626 percentage points** (**65.899% relative**).

The paper/README phrase “up to 54%” is a best-case headline, not the mean of
this particular public trace. The script reports both means and the exact
artifact row counts so those claims are not conflated.

## Compatibility fixes made in this clone

The current upstream commit is not directly runnable against current package
versions. The local reproduction patch is intentionally small and documented
in the diff:

- restore `get_benchmark_names()` and initialize the fake-backend list that
  are absent/broken in the checked-out commit;
- adapt MQT Bench calls to its 2.x `BenchmarkLevel.INDEP` API;
- patch both Runtime `FakeBackendV2` and `GenericBackendV2` with the queue
  methods expected by Qonductor, and add a `configuration().max_shots`
  fallback;
- avoid nested process pools for a one-job smoke test with
  `QONDUCTOR_TRANSPILE_PROCESSES=1`, while retaining the parallel default for
  larger runs;
- treat missing calibration entries on modern fake backends as unknown
  (neutral fidelity factor) instead of aborting the complete offline run;
- tolerate the removed `RuntimeJob` symbol and the scikit-learn 1.3 model
  serialized in the repository.

These changes do not alter the public CSV values; they only make the source
analysis and a small fake-backend scheduler path executable today.

## Boundaries and failure modes

1. The original live experiment (more than 7,000 real IBM runs) cannot be
   recollected here without an IBM Quantum account, backend availability, and
   the authors' exact historical service state. The supplied archive is the
   reproducible source for the numeric results above.
2. `qiskit_ibm_provider`'s legacy `ProviderV1`/pass-manager path is incompatible
   with Qiskit 2.x in this environment. The public estimator CSV reproduction
   does not depend on that path; a historical-version environment is needed
   to reproduce it exactly.
3. Generated PDFs can differ byte-for-byte from the authors' figures because
   Matplotlib/Seaborn versions change layout and anti-aliasing. The underlying
   CSV-derived metrics and row counts are deterministic.
4. Qonductor has distinct target meanings: circuit execution estimates,
   fidelity proxies, Pareto resource-runtime, and workflow/job JCT/utilization.
   They are not collapsed into one universal `runtime_seconds` target. This
   source-specific meaning must also be retained when these artifacts are
   integrated into the broader Quantum Runtime Estimator.

## Files added for this reproduction

These files live in the patched Qonductor checkout. Their committed fixtures
are in this capsule's `artifacts/qonductor/` directory.

- `experiments/reproduce_public_results.py`
- `experiments/fake_scheduler_smoke.py`
- `results/reproduction_metrics.json`
- `plots/reproduced/`
