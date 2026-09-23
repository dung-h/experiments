# Family-aware paper packaging

Finding recorded: 2026-09-18.
Packaged: 2026-09-23.

This folder is Cluster 3 of
[`docs/FINDINGS_CLUSTERS.md`](../../../docs/FINDINGS_CLUSTERS.md).
It stores the public-artifact reconstruction of
[arXiv:2606.11620](https://arxiv.org/abs/2606.11620) on the iQuHACK 2026
Quantum Rings 144-row table. It is not Cluster 2 (Spirit Sprinters /
SoftLocked checkpoints) and it is not a Quantum Rings SDK rerun.

## What to read

1. [REPLICATION_SUMMARY_REPORT.md](REPLICATION_SUMMARY_REPORT.md) — reporting path
2. [REPLICATION_REPORT.md](REPLICATION_REPORT.md) — protocol gaps versus the paper
3. [PROTOCOL_AUDIT_REPORT.md](PROTOCOL_AUDIT_REPORT.md) — tree baselines and context reversal
4. [MQT_FAMILY_PRETRAINING_REPORT.md](MQT_FAMILY_PRETRAINING_REPORT.md) — frozen public MQT classifier

Row-level JSON: `paper_replication.json`, `paper_replication_paper.json`,
`paper_replication_challenge.json`, `protocol_audit.json`,
`mqt_pretrained_family_replication.json`.

The copied reports still mention paths inside the original research checkout
(`experiments/replicate_paper.py`, `data/hackathon_public.json`). In this
capsule the scripts live under
[`tracks/quantum_rings/family_aware_paper/experiments/`](../../../tracks/quantum_rings/family_aware_paper/experiments/)
and the public JSON/QASM are obtained by cloning the official challenge tree.

## Data that is not copied here

- Official 36 QASM files and `data/hackathon_public.json` — clone
  `iQuHACK/2026-Quantum-Rings` at `1e2247f708136b2c6da99d7f98564ea2b15474f9`
  with `bash tracks/quantum_rings/clone_upstreams.sh work`.
- The 200 generated MQT pretraining QASM files. Rebuild with
  `generate_mqt_family_pretraining.py` if needed. The frozen classifier used
  for the packaged integration result is in `mqt_family_classifier/`.

## Reproduce from this capsule

```bash
bash tracks/quantum_rings/clone_upstreams.sh work
export QUANTUM_RINGS_ROOT="$PWD/work/quantum_rings_challenge"
mkdir -p "$QUANTUM_RINGS_ROOT/results" "$QUANTUM_RINGS_ROOT/experiments"
python tracks/quantum_rings/family_aware_paper/experiments/replicate_paper.py \
  --target both --seed 0 \
  --output "$QUANTUM_RINGS_ROOT/results/paper_replication.json"
python tracks/quantum_rings/family_aware_paper/experiments/protocol_audit.py \
  --target both --seed 0 \
  --output "$QUANTUM_RINGS_ROOT/results/protocol_audit.json"
```

Use the pin in `experiments/requirements-reproduction.txt`. Compare outputs
to the JSON files in this folder. Do not compare them to the paper PDF as if
the private 0.75 forward labels had been recovered.

`python3 scripts/verify_artifacts.py` checks that these reports and headline
numbers are present. It does not retrain the MLP or call the Quantum Rings SDK.
