# Quantum Rings / iQuHACK 2026 track

This track is an independent audit of the public iQuHACK 2026 Quantum Rings
challenge and of two public solution artifacts: Spirit Sprinters (official
winner) and SoftLocked. It is not a Quantum Rings SDK rerun, not a Ma–Li
QPU estimator, and not pooled with cuTensorNet or dense-statevector clocks.

Finding date for the artifact audit: 2026-09-18.
Packaged into this repository: 2026-09-23.

Canonical report: [`../../artifacts/quantum_rings/REPORT.md`](../../artifacts/quantum_rings/REPORT.md).
Headline numbers: [`../../artifacts/quantum_rings/summary.json`](../../artifacts/quantum_rings/summary.json).

## What the contest asks

Given an OpenQASM 2.0 circuit and an execution context, predict:

1. the minimum truncation-threshold ladder rung that meets mirror fidelity
   0.99;
2. the 10,000-shot forward wall time at the selected threshold.

The four public contexts are CPU/GPU crossed with single/double precision
(36 circuits × 4 = 144 labelled rows). Holdout QASM is private.

## Machine-profile gap

The public labels expose only `backend ∈ {CPU, GPU}` and
`precision ∈ {single, double}`. They do not name the host CPU, GPU model,
core count, DRAM, VRAM, OS, or the Quantum Rings SDK version that produced
the times. Per-run `peak_rss_mb` is process RSS, not a machine specification.
A hardware-conditioned runtime estimator cannot be calibrated from this
table. CPU versus GPU is a categorical tag, not a device profile.

## Clone pinned third-party trees

Do not copy those repositories into this capsule. Clone them on demand:

```bash
bash tracks/quantum_rings/clone_upstreams.sh work
```

Pins:

| Tree | URL | Commit |
|---|---|---|
| Official challenge (public JSON/QASM) | https://github.com/iQuHACK/2026-Quantum-Rings | `1e2247f708136b2c6da99d7f98564ea2b15474f9` |
| Spirit Sprinters | https://github.com/woody-hulse/quantum-rings | `14b5ea14dea8f6b55edf632030f5c022e8ad7373` |
| SoftLocked | https://github.com/SoftLocked/2026-Quantum-Rings | `da13c074d68c2fe3a1053c751234ee0916d73739` |

The official challenge remote is read-only for this project. Local research
commits on a private checkout of that tree are not pushed upstream.

## Boundary

- No Quantum Rings credentialed simulator rerun.
- No claim that Spirit Sprinters generalises to a hidden holdout; the 144-row
  score is a public-artifact cross-check.
- SoftLocked's saved models are valid on *their* `training_data.csv` and are
  a domain-shift failure on the public 0.99-forward table.
- Bloch, Rishivarshil and Hazel are not baselines in this track.
