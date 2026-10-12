> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../docs/thesis/README.md).

# Qonductor logical–physical consistency: 32-pair review

Five immutable pilot results plus 27 new authorized pairs. No circuit regeneration, retrospective mapping choice, dataset change or training.

## Coverage and results

Coverage: **32/32 unique physical–recipe pairs**, **11/11 logical recipes**, **230/230 linked observations**. Observation coverage is a mapping count, not 230 independent equivalence tests.

DD prepared-state proofs: **32/32**. Zero-input measured distributions: **32/32** agree within 1e−8. Maximum total variation: 9.44e-16.

DD full-unitary under the declared same input/output mapping: **8 proofs**, **24 counterexamples**. Historical initial layouts remain unknown. These failures refute that fixed-map hypothesis, not zero-input agreement.

ZX prepared-state results: **4 proofs**, **28 inconclusive**. Native statuses are retained; `probably_not_equivalent` is never relabeled as a counterexample.

| Family | Unique pairs | Linked observations | DD prepared-state proofs | DD full-unitary proofs / counterexamples | ZX prepared-state proofs / inconclusive | Zero-input measured agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dj | 10 | 48 | 10 | 1 / 9 | 1 / 9 | 10 |
| ghz | 10 | 113 | 10 | 3 / 7 | 3 / 7 | 10 |
| qftentangled | 3 | 7 | 3 | 0 / 3 | 0 / 3 | 3 |
| qpeexact | 1 | 5 | 1 | 0 / 1 | 0 / 1 | 1 |
| wstate | 8 | 57 | 8 | 4 / 4 | 0 / 8 | 8 |

## Pair-level evidence

Full SHA-256, every observation ID, measurement maps, native checker outputs and traces are in results.json (historical source; not bundled), frozen inventory (historical source; not bundled) and the unchanged pilot/new attempts. In the map, keys are physical wires and values logical wires. Measured links come from classical destinations. A single unmeasured wire, when present, is uniquely completed. Initial layout is not documented. The full-unitary test assumes the same map at input; prepared-state checks do not require that historical input-map hypothesis.

| Pair / physical hash prefix | Family | Logical / physical allocated / active width | Observations | Fixed output map | DD full / prepared | ZX full / prepared | Zero-input TV | Unit counterexample basis |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | --- |
| ghz4 / `2ca254b19943` | ghz | 4 / 5 / 4 | 5 | 0→3, 1→2, 3→1, 4→0 | PROVED / PROVED | PROVED / PROVED | 0 | none |
| wstate5 / `8c7ba1a1dccb` | wstate | 5 / 7 / 5 | 5 | 1→3, 2→4, 3→2, 5→1, 6→0 | PROVED / PROVED | INCONCLUSIVE / INCONCLUSIVE | 3.19e-16 | none |
| qftentangled5 / `d0352ec5ec6f` | qftentangled | 5 / 5 / 5 | 4 | 0→3, 1→1, 2→4, 3→0, 4→2 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 3.4e-16 | 1 (computational_basis) |
| qpeexact4 / `bc316680dc84` | qpeexact | 4 / 7 / 4 | 5 | 0→3, 1→2, 2→0, 3→1 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 2.3e-31 | 1 (computational_basis) |
| dj7 / `5c64fa25d216` | dj | 7 / 7 / 7 | 9 | 0→2, 1→1, 2→0, 3→4, 4→3, 5→6, 6→5 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 9.44e-16 | 2 (computational_basis) |
| dj3_080ee572965f / `080ee572965f` | dj | 3 / 7 / 3 | 10 | 3→0, 5→2, 6→1 | PROVED / PROVED | PROVED / PROVED | 2.78e-16 | none |
| dj5_16e03b6cd8a2 / `16e03b6cd8a2` | dj | 5 / 5 / 5 | 6 | 0→1, 1→0, 2→2, 3→4, 4→3 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 6.11e-16 | 1 (computational_basis) |
| dj5_577ad8c78e0b / `577ad8c78e0b` | dj | 5 / 5 / 5 | 2 | 0→2, 1→0, 2→1, 3→4, 4→3 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 3.33e-16 | 1 (computational_basis) |
| dj5_9b7b5fcc0bdb / `9b7b5fcc0bdb` | dj | 5 / 5 / 5 | 6 | 0→3, 1→4, 2→0, 3→2, 4→1 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 5.55e-16 | 2 (computational_basis) |
| dj5_a58338654747 / `a58338654747` | dj | 5 / 5 / 5 | 2 | 0→0, 1→2, 2→1, 3→4, 4→3 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 6.11e-16 | 2 (computational_basis) |
| dj5_aac1529295d8 / `aac1529295d8` | dj | 5 / 5 / 5 | 3 | 0→1, 1→2, 2→4, 3→3, 4→0 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 6.11e-16 | 2 (computational_basis) |
| dj5_aba2a495c06c / `aba2a495c06c` | dj | 5 / 5 / 5 | 4 | 0→0, 1→3, 2→1, 3→4, 4→2 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 6.11e-16 | 8 (computational_basis) |
| dj5_bc31f70230b8 / `bc31f70230b8` | dj | 5 / 5 / 5 | 1 | 0→3, 1→4, 2→1, 3→2, 4→0 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 5.55e-16 | 2 (computational_basis) |
| dj5_ee978197d5b1 / `ee978197d5b1` | dj | 5 / 7 / 5 | 5 | 0→2, 1→4, 2→1, 3→3, 5→0 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 3.33e-16 | 1 (computational_basis) |
| ghz5_502d1f1e5476 / `502d1f1e5476` | ghz | 5 / 5 / 5 | 3 | 0→3, 1→2, 2→4, 3→1, 4→0 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 1.37e-32 | 4 (computational_basis) |
| ghz5_5ee8af668448 / `5ee8af668448` | ghz | 5 / 5 / 5 | 17 | 0→4, 1→2, 2→3, 3→1, 4→0 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 3.08e-33 | 4 (computational_basis) |
| ghz5_97d65048c07d / `97d65048c07d` | ghz | 5 / 5 / 5 | 6 | 0→4, 1→3, 2→2, 3→1, 4→0 | PROVED / PROVED | PROVED / PROVED | 0 | none |
| ghz5_bfec24852c7d / `bfec24852c7d` | ghz | 5 / 5 / 5 | 2 | 0→2, 1→1, 2→0, 3→3, 4→4 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 1.37e-32 | 2 (computational_basis) |
| ghz5_c2604df908cb / `c2604df908cb` | ghz | 5 / 5 / 5 | 9 | 0→0, 1→1, 2→2, 3→3, 4→4 | PROVED / PROVED | PROVED / PROVED | 0 | none |
| ghz5_fd8e223cc68a / `fd8e223cc68a` | ghz | 5 / 5 / 5 | 25 | 0→0, 1→1, 2→2, 3→3, 4→4 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 1.11e-16 | 2 (computational_basis) |
| ghz7_16014940b98c / `16014940b98c` | ghz | 7 / 7 / 7 | 13 | 0→0, 1→1, 2→2, 3→3, 4→5, 5→4, 6→6 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 1.11e-16 | 2 (computational_basis) |
| ghz7_5faf82a5eac9 / `5faf82a5eac9` | ghz | 7 / 7 / 7 | 28 | 0→5, 1→4, 2→6, 3→3, 4→2, 5→0, 6→1 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 1.37e-32 | 1 (computational_basis) |
| ghz7_822d28830fbc / `822d28830fbc` | ghz | 7 / 7 / 7 | 5 | 0→6, 1→4, 2→5, 3→3, 4→1, 5→0, 6→2 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 3.08e-33 | 1 (computational_basis) |
| qftentangled5_98b35eb58df8 / `98b35eb58df8` | qftentangled | 5 / 5 / 5 | 1 | 0→3, 1→1, 2→4, 3→0, 4→2 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 3.4e-16 | 1 (computational_basis) |
| qftentangled5_b98ea8f52a1a / `b98ea8f52a1a` | qftentangled | 5 / 5 / 5 | 2 | 0→2, 1→0, 2→4, 3→1, 4→3 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 5.22e-16 | 1 (computational_basis) |
| wstate4_930d2b009f8e / `930d2b009f8e` | wstate | 4 / 7 / 4 | 5 | 1→1, 2→0, 3→2, 5→3 | PROVED / PROVED | PROVED / INCONCLUSIVE | 5.27e-16 | none |
| wstate5_01528faea64b / `01528faea64b` | wstate | 5 / 5 / 5 | 2 | 0→2, 1→1, 2→0, 3→3, 4→4 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 3.19e-16 | 2 (computational_basis) |
| wstate5_4d7ea8864b37 / `4d7ea8864b37` | wstate | 5 / 5 / 5 | 25 | 0→0, 1→1, 2→2, 3→3, 4→4 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 3.19e-16 | 2 (computational_basis) |
| wstate5_87eca3a9e7d8 / `87eca3a9e7d8` | wstate | 5 / 5 / 5 | 3 | 0→4, 1→3, 2→2, 3→1, 4→0 | PROVED / PROVED | INCONCLUSIVE / INCONCLUSIVE | 3.19e-16 | none |
| wstate5_ae156e106117 / `ae156e106117` | wstate | 5 / 5 / 5 | 10 | 0→4, 1→2, 2→3, 3→1, 4→0 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 3.19e-16 | 4 (computational_basis) |
| wstate5_b873cb4c93b2 / `b873cb4c93b2` | wstate | 5 / 5 / 5 | 2 | 0→0, 1→1, 2→2, 3→3, 4→4 | PROVED / PROVED | INCONCLUSIVE / INCONCLUSIVE | 2.78e-16 | none |
| wstate7_9206e1839c61 / `9206e1839c61` | wstate | 7 / 7 / 7 | 5 | 0→6, 1→4, 2→5, 3→3, 4→2, 5→1, 6→0 | COUNTEREXAMPLE / PROVED | INCONCLUSIVE / INCONCLUSIVE | 3.89e-16 | 2 (computational_basis) |

## Scientific implications

The complete check strengthens the functional support for the existing 230-observation reconstruction-qualified stratum: all its distinct submitted physical hashes and fixed logical recipes were examined. It does not turn those recipes into exact historical logical instances. Ideal prepared states can agree despite different unitaries, gate counts, depths and runtime. Measured-output agreement is weaker still: the Z negative control preserves probabilities while changing relative phase.

All unknown initial layouts stay unknown. Active unmeasured wires are retained; only truly idle wires are removed. Terminal measurements, barriers and delays are omitted only for the noiseless unitary view. No historical duration, calibration, original DAG or noisy-channel equivalence is inferred. Full-unitary counterexamples are genuine under the assumed map; an unknown initial layout is a possible explanation, not an established cause.

DD and ZX results are independent endpoints. DD construction establishes PROVED or COUNTEREXAMPLE within declared finite-precision semantics; ZX reduction failure stays INCONCLUSIVE. Positive self/global-phase/supplied-permutation and negative X/Z controls were run again before expansion. No qualified zero-input conflict was found, and no qualification labels, observations, model inputs, splits, original predictions or historical pilot files were changed.

## Execution and preservation

MQT QCEC 3.10.2, MQT Core 3.10.1, Qiskit 2.5.2, NumPy 2.3.5, isolated Python 3.14.4. Installed RECORD pins match the pilot. One CPU worker, no GPU. New controls/pair checks: 118 attempts, 142.36 seconds total wall, peak worker RSS 227.0 MiB. Disk free after: 4.88 GiB.

Caps: 4 GiB worker address space, 150/155 s worker CPU soft/hard limits, 120 s DD and 60 s ZX subprocess wall per pair across both semantics, 1,200 s total wall, 3 GiB disk floor, 100 MiB output ceiling. No budget was raised. Input and pilot hashes matched before and after execution. Source binding and family/pair/status/coverage reconciliation passed. Public release remains paused.

Preparation: `.venv-qcec/bin/python -B benchmark_v1/scripts/validate_logical_physical_expansion.py --stage prepare`; execution: same command with `--stage execute`; saved-result summary: `python3 -B benchmark_v1/scripts/summarize_logical_physical_equivalence.py`. The runner refuses to overwrite or rerun existing evidence. Source circuits/recipes must remain external to public distribution.

[QCEC semantics](https://mqt.readthedocs.io/projects/qcec/en/stable/equivalence_checking.html), [DD paper](https://arxiv.org/abs/2004.08420), [ZX paper](https://arxiv.org/abs/2208.12820).
