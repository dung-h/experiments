# CUDA-Q MPS runtime/feasibility pilot

**Run date:** 21 September 2026  
**CUDA-Q:** 0.15.1, revision `aca5853a76d499ecc3d5f97c2e06163ae99d9c75`  
**GPU:** NVIDIA GeForce RTX 5070 Ti, 16,303 MiB  
**Status:** local CUDA-Q MPS simulator target; not QPU/cloud runtime

## Research question

Can a pre-run feature block predict MPS execution cost when the bond cap and
cutoff are declared before execution? The pilot also checks whether the chosen
bond cap is sufficient for a high-fidelity result.

The MPS target exposes its returned tensors through CUDA-Q's `State` API. We
therefore record `observed_max_bond`, number of tensor elements and tensor
storage after the run. These fields are diagnostics/labels, not pre-run
features. The estimator never uses them as inputs.

## Protocol

The runner is
[`run_cudaq_mps_matrix.py`](../../../experiments/cudaq_runtime/run_cudaq_mps_matrix.py).
One process/run uses one predeclared `CUDAQ_MPS_MAX_BOND` value; this avoids
silently changing MPS configuration after CUDA-Q target initialisation.

- 48 rows: 4 families × 3 widths × 4 bond caps;
- widths: 16, 20 and 24;
- families: GHZ, HEA, QAOA cycle and deterministic random brickwork;
- caps: χ = 2, 4, 8 and 16;
- precision: FP64; absolute cutoff `1e-12`; SVD algorithm `gesvdj`;
- 32 shots, one warm-up and two measured warm calls;
- both `get_state` and `sample` are timed separately;
- dense-reference fidelity is computed only at width 16 to keep the reference
  comparison bounded.

## Fidelity/bond-cap result at width 16

| Cap χ | Minimum fidelity | Mean fidelity | Maximum observed bond |
|---:|---:|---:|---:|
| 2 | 0.6494 | 0.8242 | 2 |
| 4 | 0.99999 | 1.0000 | 4 |
| 8 | 1.0000 | 1.0000 | 8 |
| 16 | 1.0000 | 1.0000 | 8 |

At χ=2, the difficult families degrade substantially: HEA fidelity is
0.6494, QAOA 0.8804 and random brickwork 0.7670. At χ≥4, all width-16
families reach at least 0.99999 in this deterministic canary. This is not a
universal threshold; it is evidence that fidelity is family- and circuit-
dependent, so a fixed bond cap cannot be assumed safe.

## Runtime estimator evaluation

[`evaluate_cudaq_mps_matrix.py`](../../../experiments/cudaq_runtime/evaluate_cudaq_mps_matrix.py)
uses only pre-run fields: width, logical depth, gate counts, two-qubit count,
analytical dense-state bytes, configured bond cap and family. It excludes
observed bond, tensor storage and fidelity. It evaluates log-runtime median,
Ridge and HistGradientBoosting under grouped-circuit, family-held-out,
bond-cap-held-out and width-held-out splits.

The detailed outputs are:

- [`mps_matrix.csv`](mps_matrix.csv)
- [`mps_runtime_summary.csv`](evaluation/mps_runtime_summary.csv)
- [`mps_runtime_metrics.csv`](evaluation/mps_runtime_metrics.csv)
- [`mps_runtime_evaluation.json`](evaluation/mps_runtime_evaluation.json)

Selected Ridge log-R² values:

| Target | Grouped circuit | Family held-out | Bond-cap held-out | Width held-out |
|---|---:|---:|---:|---:|
| `state_warm_median_s` | 0.901 | −685.422 | 0.865 | 0.861 |
| `sample_warm_median_s` | 0.743 | −0.213 | 0.956 | −1.189 |

The large negative family-held-out state score and negative sample width score
are meaningful warnings, not numerical success. The current pre-run features
interpolate the controlled grid and transfer across cap values reasonably, but
they do not establish family- or width-generalisation. The family split is
especially hard because the same width has very different contraction/MPS
costs across families.

## What this establishes

1. CUDA-Q MPS is executable locally and exposes enough tensor metadata to
   measure post-run bond/storage labels without dense-state materialisation.
2. Runtime and fidelity are separate targets. A configuration can finish
   quickly while returning a low-fidelity truncated state.
3. Configured bond cap is not a substitute for observed bond demand; χ=2 is
   unsafe for several families even at width 16.
4. A static runtime model needs circuit-family/entanglement descriptors; width
   and configured cap alone are not enough.

## Limits and next extension

- Fidelity labels are currently available only for width 16; larger widths
  have runtime/tensor diagnostics but no dense reference in this artifact.
- The four families are deterministic; independent random seeds are still
  required before making a generalisation claim.
- Peak allocator memory and online remaining-time telemetry are not yet
  recorded. `mps_tensor_bytes` is a tensor-storage lower bound, not peak GPU
  allocation.
- The next controlled extension should add random seeds and a depth sweep,
  then fit a two-stage feasibility/fidelity classifier plus runtime regressor.

