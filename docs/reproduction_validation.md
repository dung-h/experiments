# Clean-clone validation

## Current E1–E6 review-package clone — 2026-10-02

The current `presentation-review` candidate was cloned with `git clone
--no-local` into a fresh directory. The clone restored the three
row-partitioned CSVs, validated the complete inventory, and rebuilt the E6
Aer/MPS aggregate from the committed predictions. The aggregate output was
byte-identical to its committed artifact.

| Check | Result |
| --- | --- |
| Release inventory | 1,467 tracked paths; 1,469 hashes including 3 materialized CSVs; PASS |
| E6 predictor aggregate | 150 Aer hashes, 27 Aer methods, 150 assigned MPS hashes (144 finite; 142 quality-pass; 6 unavailable); byte-identical PASS |
| Full `benchmark_v1/tests` suite | 155 passed in 16.59 s |
| Historical artifact check | PASS |
| Clone worktree after restore and verification | Clean |

The clone used the documented Python 3.10.21 verification interpreter from
the local verification environment with `PYTHONPATH` and `PYTHONHOME` unset.
The dependency lock itself had already passed in the candidate workspace; a
fresh-install audit of that lock is a separate environment-provisioning check.
No model fitting, CUDA workload, simulator timing, or QPU access occurred.

The package intentionally omits E4 QPY/QASM replay files and graph dumps
because the aggregate and table rebuild do not consume them. The retained
source-DAG topology is required by static materialization and is included.
This validates the review-package reconstruction path, not end-to-end
retraining from all original source data or authority to make a public
release.

## GitHub review-branch clone — 2026-10-02

The author requested publication of `benchmark-review`. Commit
`66a680bf4328524668ffa8c615025cb0b6ee45ae` was pushed to
`https://github.com/dung-h/experiments.git` and cloned from GitHub, not from
the source workspace. `main` was not changed. No release tag or repository
license was created.

Three frozen CSVs are distributed as nine row-aligned parts, each below
48 MiB. `python3 scripts/materialize_csv_parts.py` restored all three at
their original paths with matching sizes and SHA-256 hashes. Git LFS is not
needed. The original tables contain 210,408, 184,107 and 184,107 data rows;
partitioning does not change the benchmark's 8,767-observation population.

The clone was checked using a fresh Python 3.10.21 environment and the
committed verification requirements:

| Check | Result |
| --- | --- |
| Inventory | 1,318 tracked paths and 3 ignored, reassembled CSVs; 1,320 file hashes verified |
| CSV restore regression tests | Multiline cells, CRLF, missing final newline, corrupted part rejection and refusal to overwrite mismatched data covered |
| Full included test suite | 42 passed in 12.76 s |
| Historical artifact check | PASS |
| Published scorecard validator | PASS; 24 QPU variants, 72 source pairs, 162 supplementary Aer rows |
| Simulator table rebuild | All 7 non-manifest files byte-identical |
| QPU rebuild | `attempts.csv` and the 3 metric tables byte-identical |
| Scorecard rebuild | All 22 non-manifest files byte-identical |
| Git status after restore, tests and rebuilds | Clean |

Commands are in [reproduction.md](reproduction.md). Validation used no
original-workspace data fallback, training, simulator timing or QPU access.
The included data supports metric rebuilding, not complete original-dataset
retraining. The scientific gaps and external-source limitations described
in the methodology remain unchanged. `CURRENT.json` and the C4 receipt
retain the hashes recorded below.

## Earlier local-clone check

**Status: internal reproduction package passed. Not a public-release approval.**

On 2026-10-02, a fresh local Git clone was made with `git clone --no-hardlinks`
from candidate commit `262cb726175bd3e11e1c9756f6a8986ce474d691`. The complete
copy/paste procedure is in [reproduction.md](reproduction.md). No source
workspace files, canonical raw rows, predictions, split manifests, `CURRENT`,
or C4 receipt were rewritten.

## Environment and checks

| Check | Result |
| --- | --- |
| Python | 3.10.21 |
| Verification lock | NumPy 2.2.6; pandas 2.3.3; SciPy 1.15.3; scikit-learn 1.7.2; pytest 9.1.1 |
| Release inventory | 1,308 tracked paths; 1,307 file hashes verified; 481,905,261 bytes; PASS |
| Historical capsule integrity | `scripts/verify_artifacts.py`: PASS |
| Published scorecard validation | 24 QPU variants, 72 source pairs, 162 supplementary Aer rows; Maestro remains `pilot_gate_failed` with no score; PASS |
| Simulator reader tables | Rebuilt and validated; all 7 non-manifest files byte-identical to the published reader table pack |
| QPU aggregation | 24 methods, 72 source-local pairs, 36 coverage-asymmetric pairs; `attempts.csv` and 3 metric CSVs byte-identical |
| Two-domain scorecard | Rebuilt with `BUILT_FROM_PINNED_INPUTS_MAESTRO_PILOT_FAILED`; all 22 non-manifest files byte-identical |
| `benchmark_v1/tests` | 38 passed |
| Clean-clone worktree after checks | Clean; environments and rebuilds are ignored under the documented local paths |

The QPU aggregation ran CPU-only with `CUDA_VISIBLE_DEVICES=''` and one thread
for OMP, OpenBLAS, MKL and NumExpr. No model training, simulator timing, GPU
workload or QPU submission was performed. Exact commands and output paths are
listed in [reproduction.md](reproduction.md); generated manifests can differ
by output path, so the byte comparisons intentionally exclude manifests.

The validation confirmed these immutable pins:

- `benchmark_v1/registry/CURRENT.json`: `8a4ba7288e46324c49e290a1244c518b9f01b73a7f27a9b4a79984a13135c9e3`
- C4 `receipt.json`: `e6e30da64c71ae79c75d1382779fe4910972992986a57368b49a5c0802c5a059`

## What this proves—and does not prove

The tested workflow validates the Git payload and rebuilds published metrics
from frozen features, predictions, attempts and simulator measurements. It
does not reproduce original-data extraction, retrain every paper method, or
repeat simulator measurements. The complete Ma–Li and Qonductor source QASM
collections are not bundled; QPack's optimized-angle submitted circuits and
full source execution payload were not recovered. The corresponding
reconstruction and availability limits are stated in
[methodology.md](methodology.md) and [release_inventory.csv](release_inventory.csv).

This was a local clone of the internal candidate, not a clone from a public
remote. The candidate contains 1,308 files totaling about 482 MB; its three
largest CSVs are about 87–167 MB each. Remote hosting/large-file delivery,
author-selected `LICENSE` and `CITATION.cff`, and third-party data permissions
remain unresolved under [S56](../benchmark_v1/S56_WAVE5_RELEASE_LICENSE_AND_PROVENANCE_DECISION_V1.md).
Therefore this PASS is evidence for internal clean-clone reproducibility of
the derived tables, not permission to publish or a claim of end-to-end source
dataset reproducibility.

The clone exposed two packaging issues before this pass: the immutable
scorecard validator needs its separately pinned historical builder copy at
the path recorded by the manifest, and the reproduction command must write
under the builder's supported `work/repo_finalization/rebuilds/` directory.
Both are now explicit in the payload inventory and reproduction guide; no
scientific data, prediction or authority manifest was changed.
