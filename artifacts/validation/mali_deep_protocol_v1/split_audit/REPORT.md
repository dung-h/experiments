# Ma--Li leakage-safe split audit

Generated: **2026-09-21**

This manifest uses SHA-256 of raw QASM bytes as circuit identity. It is
a data/split audit; it does not retrain a model or reconstruct a historical
IBM calibration snapshot.

## Inventory

- label rows: **340** (osaka=192, kyoto=148)
- exact logical-QASM hashes: **170**
- hashes present on both backends: **130**
- distinct `(QASM hash, backend)` cells: **300**
- duplicate cells under filename aliases: **40**
- family connected components: **9**

## Primary grouped folds

All backend copies and filename aliases of one QASM hash stay in the
same outer fold. The fold number is deterministic for the recorded seed.

| Fold | QASM groups | Rows | Osaka | Kyoto |
|---:|---:|---:|---:|---:|
| 1 | 34 | 65 | 37 | 28 |
| 2 | 34 | 68 | 40 | 28 |
| 3 | 34 | 74 | 40 | 34 |
| 4 | 34 | 67 | 38 | 29 |
| 5 | 34 | 66 | 37 | 29 |

## Strict backend + unseen-QASM diagnostic

For each held backend and grouped fold, test rows are held-backend rows
in that QASM fold. Training rows are the other backend with all QASM
groups in the test fold removed. This is a two-backend diagnostic, not
broad vendor generalization.

| Held backend | Fold | Test rows | Train rows | Hash overlap |
|---|---:|---:|---:|---:|
| osaka | 1 | 37 | 120 | 0 |
| osaka | 2 | 40 | 120 | 0 |
| osaka | 3 | 40 | 114 | 0 |
| osaka | 4 | 38 | 119 | 0 |
| osaka | 5 | 37 | 119 | 0 |
| kyoto | 1 | 28 | 155 | 0 |
| kyoto | 2 | 28 | 152 | 0 |
| kyoto | 3 | 34 | 152 | 0 |
| kyoto | 4 | 29 | 154 | 0 |
| kyoto | 5 | 29 | 155 | 0 |

## Family components

Filename families are connected through exact QASM hashes before
holdout. This prevents a byte-identical circuit named under two family
tokens from crossing a family split.

| Component | Families | Hashes | Rows |
|---|---|---:|---:|
| `qft` | `qft` | 26 | 46 |
| `qftentangled` | `qftentangled` | 26 | 47 |
| `qnn` | `qnn` | 14 | 26 |
| `qpeexact` | `qpeexact` | 22 | 35 |
| `qpeinexact` | `qpeinexact` | 22 | 36 |
| `qwalk_noancilla` | `qwalk_noancilla` | 1 | 2 |
| `random` | `random` | 16 | 30 |
| `realamprandom+twolocalrandom` | `realamprandom+twolocalrandom` | 22 | 80 |
| `su2random` | `su2random` | 21 | 38 |

## Interpretation

The old row-level split can measure label adaptation when the same
logical circuit appears on both devices, but it cannot support an
unseen-circuit claim. Use `row_manifest.csv` to drive every later model
evaluation. Simulator pretraining must also exclude outer-test QASM
hashes when reporting strict all-domain transfer.
