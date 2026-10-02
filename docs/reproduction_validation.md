# Clean-clone validation

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
