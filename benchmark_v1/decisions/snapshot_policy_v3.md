# S66 — Snapshot policy V3

**Decision date:** 2026-09-29  
**Status:** signed; registry materialization and target preflight pending  
**Scope:** archived real-QPU analytical estimators and any frozen
hardware-context feature.  This decision does not alter local simulator
measurement contexts.

## Rule

Every snapshot-dependent attempt must cite one record from the V3 snapshot
registry.  That record, not a backend name alone, is the hardware-context
input.  An attempt without its required record is terminal `unavailable`.

The registry must retain, where the source exposes them: backend canonical
name and aliases; FakeBackend class; package name/version; Qiskit version;
environment lock hash; configuration, properties and defaults JSON path/hash;
Target construction method; captured/update date; instruction-duration map;
operation errors; T1/T2/readout fields; target operation coverage; and the
snapshot tier.  Absence of a field is recorded as absence, not filled from
another backend.

The inventory universe is `Belem`, `Brisbane`, `Jakarta`, `Kolkata`, `Kyoto`,
`Lagos`, `Lima`, `Manila`, `Nairobi`, `Osaka`, `Perth`, and `Quito`.  Canonical
backend identity normalizes the public `ibm_*` / legacy `ibmq_*` spelling; the
unmodified observed spelling remains in the raw ledger.

## Snapshot tiers

| Tier | Meaning | Allowed reporting role |
| --- | --- | --- |
| 1 | same backend/version and dated target or calibration aligned to job day | narrow historical proxy claim, if all method inputs exist |
| 2 | same backend with a dated snapshot within 62 days | historical approximation, separately stratified |
| 3 | same backend but stale, undated relative to label, or version-unmatched | nominal/sensitivity only; never job-day calibration |
| 4 | declared architecture proxy from another backend | sensitivity/ranking only |
| 5 | no admissible method input | explicit unavailable |

Ma--Li and QPack lack archived label timestamps.  A same-backend public
snapshot may therefore be tier 3 and may support an explicitly nominal
sensitivity, but cannot become a historical calibration claim.  Qonductor
rows retain their timestamps; a later snapshot remains tier 3 unless a
row-aligned version/date rule proves otherwise.

## Consequences for named methods

- Qiskit `estimate_duration`, QCRE snapshot critical path, and the
  one-circuit Hyb-HANAS adaptation may emit a raw *proxy* at tiers 1--3 only
  when their full input set is present.  Tier 3 outputs are diagnostic and
  stratified by tier.
- `scholten_original_style` accepts tier-1 historical `CLOPS_v` only.
  `CLOPS_h`, T1/T2, duration, error rate, width and depth are not conversions
  to `CLOPS_v`.
- `scholten_clops_h_denominator_substitution_sensitivity` is governed by the
  signed Ma--Li tier-3 amendment and remains a separate diagnostic method ID.
- No snapshot field, tier, or target construction can read `target_seconds`,
  a held-out label, or an observed-runtime aggregate.

## Frozen environment for C133--C135

The registry/preflight environment is
`/home/server/Documents/.venv-qonductor/bin/python`, with Qiskit `2.5.2` and
`qiskit-ibm-runtime 0.49.0`.  C133 must capture `pip freeze`, interpreter
path/version, CUDA visibility and host fingerprint as immutable evidence.
If this environment does not expose the pinned snapshot faithfully, C133 must
report the incompatibility; it must not silently substitute the latest
FakeBackend.

## Supersession

This decision supersedes prior snapshot-policy wording only where it described
an incomplete registry as a terminal scientific absence.  It does **not**
overwrite `c121_terminal_unavailable_v2`: that historical output remains
preserved and must be marked superseded in a later Wave-0 record.  It also does
not turn a nominal snapshot into historical evidence.
