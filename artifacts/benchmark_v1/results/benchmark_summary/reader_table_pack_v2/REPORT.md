# Reader table pack v2

Derived 2026-09-30T14:27:49.858354+00:00. Status: PASS.

This directory does **not** replace pack v1, the current scorecard, c66 tables,
the fidelity overlay, Wave 3 reports, the C4 receipt, or raw predictions.
Pack v1 remains a historical sibling.

## Sources

See `source_hashes.json`. Every numeric cell cites a source path and sha256.

## Counts

- view A rows: 33
- view B rows: 48
- view C rows: 106
- view D rows: 22
- blocked/unavailable/contract_only rows: 16
- scores recomputed: False
- prediction CSVs opened: False
- GPU/training started: False

## Explicit blocked cells

V3-large has no published per-source MAE and no published paired MAE versus
the unified polynomial. Those cells are blocked rather than fabricated.

V4 has no score. Ma--Li 340 / Qonductor 4482 are not 8,767 unified rows.
The three-seed Ma--Li median is NOT_AUTHORIZED.

## C4 receipt

S81 records the C4 test-count erratum. Receipt hashes were not modified.
