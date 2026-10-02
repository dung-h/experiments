# Simulator predictive-runtime figures (E6 aggregate)

Two separate figures are provided because Aer and CUDA-Q MPS have different evaluation targets and clocks. Neither chart ranks a method across those targets. They use only the frozen E6 predictive aggregate; no raw timing or retraining was performed.

## Figure 1 — Aer warm-execution paired contrasts

Files: `fig1_aer_paired_predictor_contrasts.png` (SHA-256 `bffad13adf01c040b6b7e0b171e2c91cf1a76fc96045d88f8216afe0b0763bbe`), `fig1_aer_paired_predictor_contrasts.svg` (SHA-256 `89bda42049379532bd3d9293bf031862f7153debffbca5c10332c60fc958d12f`).

Population: 150 shared circuit hashes; source_sha256 grouped bootstrap. Target clock: `aer_azizov_core_q9/warm_execution`; prediction clock: `predicted_noisy_aer_warm_execution_seconds`. Each Δ is candidate MAE minus reference MAE on the same source-hash groups; positive means the candidate has larger error. Error bars are source_sha256-grouped 10,000-replicate 95% percentile bootstrap intervals. The source-to-transpiled graph contrast excludes zero; the hybrid-to-transpiled and transpiled-GNN-to-XGBoost intervals include zero. These unadjusted intervals do not establish general superiority or transfer beyond this single local Aer target.

MAE context on 150/150 rows: source GNN 0.450 s; hybrid GNN 0.406 s; transpiled GNN 0.301 s; transpiled XGBoost 0.297 s.

## Figure 2 — fixed-χ16 MPS warm-state-execution paired contrasts

Files: `fig2_mps_paired_predictor_contrasts.png` (SHA-256 `332cc2d12de8f9242fd5ca00bad36dc46a999f10418c9d453ae52de7400e56ad`), `fig2_mps_paired_predictor_contrasts.svg` (SHA-256 `648e0c10bee1a803396fd9693e141e76aa926a583f7dc5100cd54ee0e8343761`).

Target clock: `cudaq_mps_fp64_bond16_warm_state_execution_seconds`; prediction clock: `predicted_cudaq_mps_fp64_bond16_warm_state_execution_seconds`. The left panel has 144 finite paired rows; the right is the prespecified 142-row fidelity-pass subset (`fidelity ≥ 0.99`). Two finite but quality-failed target rows occur only in the all-finite population. Δ is E5 family-residual MAE minus each named reference; positive means E5 error is larger. Intervals are unadjusted source_sha256-grouped 10,000-replicate 95% percentile bootstrap intervals. Family-residual vs family-agnostic includes zero in both panels; residual-vs-E3-graph is positive. These are comparisons within the fixed MPS configuration, not with Aer.

All-finite MAE context: E3 graph 0.302 s; E5 family residual 1.001 s; E5 family agnostic 1.081 s; E3 Ridge 1.229 s.

## Source pins

The aggregate manifest validates each plotted CSV hash. The source hash registry also pins the underlying E6 data/code/environment inputs.
- `artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/aggregate_manifest.json` — SHA-256 `701007ea8d38899fcc3bcd5c51a11e236b71b0423faad67e7ca75130bd6d9a96`
- `artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/source_hashes.json` — SHA-256 `2e81047fda9c2fe37967763e440a5d330b45f5ba60a07c5a973bbb3d3f2085c2`
- `artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/aer_azizov_hash_metrics.csv` — SHA-256 `16a6652056a4cc989b9d263a42bd9d97d4615460a13b32a63644c34ee3302b6a`
- `artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/aer_azizov_paired_bootstrap.csv` — SHA-256 `771cb644ecf2fabe8986164c0379ada85fab4f08be9a363eeac8ee5b8a85a336`
- `artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/mps_fixed_chi16_hash_metrics.csv` — SHA-256 `710718e5b80deb42b1b993bc9b8327f0a8e0f1bcf0441f3bdc0467e10639c09e`
- `artifacts/benchmark_v3/simulator/predictive_runtime_aggregate_v1/mps_fixed_chi16_paired_bootstrap.csv` — SHA-256 `c6db61e92d09054485684ca768ef12eb47e9eab0a098f9f55c49f23d1b3f1cae`
- `presentations/source/plot_predictive_figures_v2.py` — SHA-256 `1462a8c836be7cb5836ff92549f5c1aef80c7c0fe664c4a2432d7d0333c87de7`

## Reproduction

From repository root, with the packages in `presentations/requirements-figures.txt` installed:

```bash
python presentations/source/plot_predictive_figures_v2.py
```

The script reads the frozen aggregate, checks its manifest pins, and writes only this directory. It does not modify benchmark data, scorecards, or `CURRENT.json`.
