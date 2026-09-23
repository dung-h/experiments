# Family-aware paper scripts

Vendored copies of the 2026-09-18 reconstruction scripts for
[arXiv:2606.11620](https://arxiv.org/abs/2606.11620). Packaged results live in
[`artifacts/quantum_rings/family_aware_paper/`](../../../artifacts/quantum_rings/family_aware_paper/).

These scripts read `data/hackathon_public.json` and `circuits/*.qasm` from
`QUANTUM_RINGS_ROOT`. Point that variable at the official challenge clone:

```bash
bash tracks/quantum_rings/clone_upstreams.sh work
export QUANTUM_RINGS_ROOT="$PWD/work/quantum_rings_challenge"
```

If the scripts are copied into `work/quantum_rings_challenge/experiments/`,
the default `ROOT` is the contest checkout and the environment variable is
optional.

Do not push this overlay to `iQuHACK/2026-Quantum-Rings`.
