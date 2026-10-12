> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Validation report

Scope: saved-evidence analysis/reporting only. Không experiment, model inference,
new calibration, QCEC, dataset/protocol/checkpoint change hoặc historical rewrite.
Assessment judgement khác mechanical verification: PARTIALLY SUFFICIENT không
được đổi thành complete chỉ vì các checks bên dưới PASS.

## Numerical/source checks

- Current cohort 4.515, source counts 340/230/3.945, groups 165 khớp frozen profile.
- Track A34 metric rows; Track B49 rows/seven predictive contexts, không cross-trackpool.
- 51 raw / 41 upstream sim / 40 parent inputs giữ đúng support evidence; RQ4 fit-dependent.
- Existing 21-case/126-cell authority giữ nguyên, no case resume hoặc new refit.
- Graph skill ratios tính lại từ frozen MAEs; paired CIs lấy đúng saved direction.
- TN core/frontier medians 3.734/4.132 lấy từ saved stratum rows; không overall median.
- Thirty parent manifests pin exact cells; elapsed timer precedes test inference.
  Cost inventory không là matched deployment timing hay campaign duration.
- New coverage 55 rows:fiveTrack A +fiveRQ×tenTrack Bcontexts, gồm seven predictive
  contexts, native TN, analog và separate historical width. Đây không 55 experiments.
- Context status labels vàallowed actions được kiểm tra riêng; negative findings
  không bị đổi thành UNANSWERED. No unavailable value imputed as zero.

Exact headline values used:

```json
{
  "graph_mae_seconds": 0.8341333281956075,
  "graph_minus_selector_ci": [
    -0.3178575659475499,
    0.45790751386189305
  ],
  "parent_recorded_loop_costs": {
    "mali_global_mlp": {
      "cells": 15,
      "max_seconds": 83.72796021199974,
      "median_seconds": 76.72213048999402,
      "min_seconds": 74.19720708300156,
      "sum_cell_seconds": 1173.3004590609926
    },
    "mali_logical_graph": {
      "cells": 15,
      "max_seconds": 1046.6947389199995,
      "median_seconds": 1008.4794922899964,
      "min_seconds": 972.5320141970005,
      "sum_cell_seconds": 15143.611649124996
    }
  },
  "skill_vs_median": 0.2671956039683391,
  "skill_vs_selector": 0.059237156805844404,
  "tn_EST_over_warm_medians": {
    "core_q2_q9": 3.7340122316333324,
    "frontier_q10_q16": 4.132415604942254
  },
  "tn_successful_session_stage_costs": {
    "end_to_end_nonempty_rows": 0,
    "t_first_contract_s": {
      "max_seconds": 0.04983689498901367,
      "median_seconds": 0.0016592960357666015,
      "min_seconds": 0.0014831039905548096,
      "n": 603
    },
    "t_network_build_s": {
      "max_seconds": 0.38716221100185066,
      "median_seconds": 0.2947701569937635,
      "min_seconds": 0.28481998200004455,
      "n": 603
    },
    "t_path_search_s": {
      "max_seconds": 58.007454769001924,
      "median_seconds": 0.012510535001638345,
      "min_seconds": 0.0008001489986781962,
      "n": 603
    },
    "t_warm_contract_s": {
      "max_seconds": 0.03995452880859375,
      "median_seconds": 0.00013760000467300415,
      "min_seconds": 3.292800113558769e-05,
      "n": 603
    }
  }
}
```

## Integrity and limits of validation

Manifest pins all directly consumed source files,30 parent manifests, reporting
builder and originalrequest. Validation rechecks each source digest and exact
byte rendering of nine requested files; links must resolve and CSV rows/columns
must replay. Existing thesis validator is also re-run against its pinned
frozen-evidence sources/receipt, not treated as new experimental reproduction.
No timestamp/elapsed from execution is relabeled inference overhead.

No new primary-score contradiction found. Existing reporting clarifications
(TNlabels and historical mean-predictions scoring) remain. Frozen contract's
historical 10/21 statements do not replace full 21-case current report. NewRQ numbering
is explicitly mapped, not silently applied to old protocols.

Notverified: author executed real tensor, universal original method fidelity,
prospective safety, isolated generation/feature causality or matched estimator
end-to-end cost. Absence of these prevents stronger claims. This validator
does not prove completeness, training reproducibility or publication approval.

Run from repo root (read-only):

```sh
PYTHONDONTWRITEBYTECODE=1 /home/server/Documents/.venv-qonductor/bin/python -B  benchmark_v1/scripts/build_research_question_closure.py
```

## Verification observed during this assessment

- Closure source/byte/numerical/link validator: PASS, 77 source pins,
  55 context rows, zero broken local links, zero authorized amendments.
- Focused tests: 36 passed in 6.25 s. Command used `python -B -m pytest -q
  -p no:cacheprovider` on `test_research_question_closure.py`,
  `test_thesis_interpretation.py`, `test_benchmark_interpretation.py` and
  `test_final_scientific_reporting.py`, with OPENBLAS/OMP threads set to 1
  and PYTHONDONTWRITEBYTECODE=1. These are reporting checks, not experiments.
- Existing thesis validator rerun: PASS, 25 reporting files and 66 source pins;
  historical thesis manifest remains
  `9a7744bae230928bde433ee0cc4064a9e4feaf89b060f5f9de1e78c0c8107c60`.
- `git diff --check`: exit 0. Existing 13 tracked modifications were already
  present before this task and were not edited. New analysis files only.

The test run above preceded this verification paragraph and final prose-spacing
edits. The same focused tests and both validators are rerun on final saved files
before handoff; the final response records that final run separately.

Sources below are exact local evidence paths/hashes, not a public-data export:

| Source | SHA-256 |
| --- | --- |
| artifacts/benchmark_deep_dive/diagnostics/tn_scaling_v1.json (historical source; not bundled) | `a9f98bde7bc68586050ff6bd0a4c36b9fca4c79c3981f9658c3fac23ba2e5fc8` |
| artifacts/benchmark_deep_dive/scripts/simulator.py (historical source; not bundled) | `75f18f5dd4df57dc66dc663f8f54579b93932ab36e260c4785552d5a2260c656` |
| [artifacts/benchmark_deep_dive/tables/analytical_controls_v1.csv](../../analysis/analytical_controls.csv) | `f293c98475b01f15a38fcf83a17bed48165a6056b63047d231b1f20c0b247f74` |
| [artifacts/benchmark_deep_dive/tables/mps_selection_safety_v1.csv](../../analysis/mps_selection_safety.csv) | `d564edf1557fbb3400bf192c6805639e9b06ee70ddc7aa76639c606496772f47` |
| [artifacts/benchmark_deep_dive/tables/pasqal_stage_quality_v1.csv](../../analysis/pasqal_stage_quality.csv) | `9a9ee910b40fd509873c2f997d7b6166402f36f42a48e4f43209305e448c8409` |
| [artifacts/benchmark_deep_dive/tables/qpu_metrics_v1.csv](../../analysis/qpu_metrics.csv) | `958552fa0d376e7c6609cfd85f51ce3d147fd74ffb239a3054db9f432659659f` |
| [artifacts/benchmark_deep_dive/tables/qpu_shared_reference_comparisons_v1.csv](../../analysis/qpu_shared_reference_comparisons.csv) | `df9d20e6b46a8192c202e95cbf68f412aec4ce51eed81b0b1ad8da71c290c2b2` |
| [artifacts/benchmark_deep_dive/tables/simulator_metrics_v1.csv](../../analysis/simulator_metrics.csv) | `b7f11383bccf7022f4b26b207cd3c95a1b830676158223d6058d2a8979fd3535` |
| [artifacts/benchmark_deep_dive/tables/simulator_paired_v1.csv](../../analysis/simulator_paired.csv) | `6958ebf05498a6aed739b10f8dff3a074a81e9bc160b226f8557d0f7ef6893cb` |
| artifacts/benchmark_deep_dive/tables/simulator_slices_v1.csv (historical source; not bundled) | `eae8d35ecda11419f4ea875dd87f893e7639cc0ad74d412209c7653c41c6b54b` |
| artifacts/benchmark_deep_dive/tables/simulator_target_width_distributions_v1.csv (historical source; not bundled) | `3d10130f456279cc32f5e280f894728c8e47a5914a5abdd6205313c09f387a51` |
| [artifacts/benchmark_deep_dive/tables/tn_same_plan_slices_v1.csv](../../analysis/tn_same_plan_slices.csv) | `0e64f0f0fee78ecc38e593c60d94c1610cff35c78bd440856ce3bcb58024f4dd` |
| [artifacts/benchmark_interpretation/v1/README.md](../../interpretation/benchmark/README.md) | `279844e48255cc346e8faaadaa9f4795836e1b328a6607d6a9830a423336d3a2` |
| [artifacts/benchmark_interpretation/v1/benchmark_question_matrix.csv](../../interpretation/benchmark/benchmark_question_matrix.csv) | `6a849690bbc195f4ea14870b4f8abfed1a76b4863b637b1f323ed2539c9f23ab` |
| [artifacts/benchmark_interpretation/v1/effective_input_widths.csv](../../interpretation/benchmark/effective_input_widths.csv) | `6a3500831aa6a2e701ae1b023377e5a6eac9367c7baa7f86aa22c95176d2480a` |
| [artifacts/benchmark_interpretation/v1/generalization_field_support.csv](../../interpretation/benchmark/generalization_field_support.csv) | `8692565e2a3bef4b5f2502cc2fd475564de463ac2c8f50b3d737dcc2c3668091` |
| artifacts/cudaq_runtime/launch_scaling_v1_20260922/analysis/REPORT.md (historical source; not bundled) | `28803af09a45b0ce6a88638a1e93ae7ed30b16685e592c4dbb85f948f5d4e38e` |
| [artifacts/final_research/rq4_report.md](../../generalization/report.md) | `0d710cb5da210eac2c30db59663aa998d04b37ecaafc23059d8a6a53f2ce8238` |
| artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011/manifest.json (historical source; not bundled) | `8bb922e5ac7c02ea30c81be6002874bce88e4876c69405a4e01aad8c806f6dbc` |
| artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011/method_metrics.csv (historical source; not bundled) | `559de395c14ee3d990e3054bcf04451e3ec9ad82ff6648d81b57d24da012e801` |
| artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011/reader_tables/cell_lineage.csv (historical source; not bundled) | `07de9dc18c2174041a841ae87f861b54a6ccc40fbd00a32e9b72f09f8bce16bd` |
| artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011/reader_tables/input_support.csv (historical source; not bundled) | `f4caaa416140cd917a319c43befcefec0bae30aecac2c4e68d7309883f28e83f` |
| artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011/reader_tables/paired_comparisons.csv (historical source; not bundled) | `11da43817bcf700aa228f17096b98cc1e6732e66360d2cdd55f200693b2f44d9` |
| artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011/reader_tables/source_metrics.csv (historical source; not bundled) | `85f87ab71dbf13d490ac68328f7950221ef24afb759b4046fedf83a61400ce5f` |
| artifacts/real_qpu/provenance_equivalence/results.json (historical source; not bundled) | `3b8a882068ddf8b8fe2652cb75fa96d136e974bb420c6e6508dadb6e5a1da66f` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_0/mali_global_mlp/seed_1234/manifest.json (historical source; not bundled) | `4c1a841683f40669f7bceeca47052174b5149abf900e1a24be938dbd75e21adb` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_0/mali_global_mlp/seed_31415/manifest.json (historical source; not bundled) | `882da6be38812437b87971a9306acd25e2111091ff7f7383fc0705b602bcecb8` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_0/mali_global_mlp/seed_42/manifest.json (historical source; not bundled) | `b360b96b443b5d7072693b0d2d5bd41e7d8dc290d83468b50e08af276682bc85` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_0/mali_logical_graph/seed_1234/manifest.json (historical source; not bundled) | `fc25126e08925dd2c449d8b0f2c00bd7a3d59acec4e1040368cdcbb6951084b0` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_0/mali_logical_graph/seed_31415/manifest.json (historical source; not bundled) | `11081570836d14212471994ce3e37405ade53f0cd25d2261523c2e4e4b6ade26` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_0/mali_logical_graph/seed_42/manifest.json (historical source; not bundled) | `70aef7f34ffc2ee7adab425760bfc66581942ee4aadac7059438d1c04f33c616` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_1/mali_global_mlp/seed_1234/manifest.json (historical source; not bundled) | `f4b7ab58fe8dfc0a9e98d6697e5ce4f21901f0a5bd0ffcbf93a452615609e781` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_1/mali_global_mlp/seed_31415/manifest.json (historical source; not bundled) | `01c8a5098b980c1b51e8985cd314d459caf25d9f9c8888d66b13675f95640613` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_1/mali_global_mlp/seed_42/manifest.json (historical source; not bundled) | `926b1c45c3f900f7d331e53f659c049aaa690d59240abde9da83b98341691c86` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_1/mali_logical_graph/seed_1234/manifest.json (historical source; not bundled) | `5de80368e360023ee10b0561cb4932e73f57d625a58aae3ef239c10532e3d403` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_1/mali_logical_graph/seed_31415/manifest.json (historical source; not bundled) | `edd6afcd51aad98636e7867279819546169ae5a7990c7ae481e41c449cf99a0a` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_1/mali_logical_graph/seed_42/manifest.json (historical source; not bundled) | `a01c46badd75cf38d0faa8e3857b6339170a5218dcffbbc08712c94db0a71169` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_2/mali_global_mlp/seed_1234/manifest.json (historical source; not bundled) | `20930135cb9f66102b6502500b5fa569c596d2d84741dfdc5b3fa5df6a6e5625` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_2/mali_global_mlp/seed_31415/manifest.json (historical source; not bundled) | `bc06a15df51dd3676bd88c443ab460aba8492c95e1ea928be6cb2f051ab8d18a` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_2/mali_global_mlp/seed_42/manifest.json (historical source; not bundled) | `34a1ed994baaf3963ad4fca10ab2d1e2ffc3feb864afddf3d4ed7eb3bf993049` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_2/mali_logical_graph/seed_1234/manifest.json (historical source; not bundled) | `409d2d646264b85b2ec629dc02f2e732f93ddd0e515eb002ea50070a90873671` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_2/mali_logical_graph/seed_31415/manifest.json (historical source; not bundled) | `0e79b49b4a9b987e531341446bfb9f3ff0f7dd99387c83b77da88edaeddbe73a` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_2/mali_logical_graph/seed_42/manifest.json (historical source; not bundled) | `c21b4eb30d7005cf01e00632584f589085148b23d88248ba2680cd07b818714f` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_3/mali_global_mlp/seed_1234/manifest.json (historical source; not bundled) | `483f012ec96a212ef5a70e7ec4004693cd0ec67a0623c133ff0e7f8248809959` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_3/mali_global_mlp/seed_31415/manifest.json (historical source; not bundled) | `08f42c5abf7f2168978d95af165b5f1d9eb38819a479971a4505ff79d9ce52cc` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_3/mali_global_mlp/seed_42/manifest.json (historical source; not bundled) | `7de16485e7326e12ceffe784bffa7c13a5511bb8b9939e3a9767f00d9cbb3195` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_3/mali_logical_graph/seed_1234/manifest.json (historical source; not bundled) | `827da65ced78a5937efd1cde45e5409934f0ce10c3b01880f121008a8aab40e3` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_3/mali_logical_graph/seed_31415/manifest.json (historical source; not bundled) | `ff8494cb97e822cceb7664f0f1de6a2e0c62004cd88aaeb6fa1deb32caea7fa7` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_3/mali_logical_graph/seed_42/manifest.json (historical source; not bundled) | `b2d51a3fe761b322d327e85d587cd8871096d582522f45a7287f3ce5d693d25d` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_4/mali_global_mlp/seed_1234/manifest.json (historical source; not bundled) | `0cd45c9e77bb8ae47a4f01e18e56aabf9ad492c9d309773d8eb2a2c61de52536` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_4/mali_global_mlp/seed_31415/manifest.json (historical source; not bundled) | `d3ae41610e15c564da5b913e02d0f43664aaeb60b85b051c2283fab9882392bc` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_4/mali_global_mlp/seed_42/manifest.json (historical source; not bundled) | `d08d031a6ada4b580d910583ed36f3cfab3f8ac32fc968db0ab3055c76101dd8` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_4/mali_logical_graph/seed_1234/manifest.json (historical source; not bundled) | `3195cf9477bcb3f49058ac223fcb644b94aa09f259dcecd6eb95f23443c83268` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_4/mali_logical_graph/seed_31415/manifest.json (historical source; not bundled) | `5396a2c868f3c7472a297c1daef15ea0012eb81a4e98a85a3584d2ed97aa5565` |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_4/mali_logical_graph/seed_42/manifest.json (historical source; not bundled) | `f1f22ad3ef565e1251cec29e9bffeca4a596f4a43e7c6fece82619be9040697d` |
| artifacts/real_qpu/qualified_logical_panel/summary/dataset_profile.json (historical source; not bundled) | `c283a7a84f6cd68c30f5416584f959a1c47a7f9608b5c102e5c9c851e7ebf7a7` |
| artifacts/real_qpu/qualified_logical_panel/summary/shared_row_comparisons.csv (historical source; not bundled) | `f620fd6cf157b8a3cb7491a6d8fe9f116c75e65e60b774101ee95bb360110919` |
| [artifacts/thesis_interpretation/v1/README.md](../../interpretation/thesis/README.md) | `0de96df9e7ec9d0b3b9c76fe7ee261aa6872cbfb4df74d503455735ae0b21c16` |
| [artifacts/thesis_interpretation/v1/estimator_fidelity_registry.md](../../interpretation/thesis/estimator_fidelity_registry.md) | `dd2589afa5117fc734d3773c6718ed2cb0fd34e8591523adae18a647e53d2a67` |
| [artifacts/thesis_interpretation/v1/evidence_to_claims.csv](../../interpretation/thesis/evidence_to_claims.csv) | `164cd56237d6c065865f6f16a169e31b749ecaaf2c146f9a9534af9b2448bb45` |
| [artifacts/thesis_interpretation/v1/final_scientific_interpretation.md](../../interpretation/thesis/final_scientific_interpretation.md) | `81a1d77d6a908998c2db54a93d6623d87cd7fed318e5639b0b330ed2d78c692c` |
| artifacts/thesis_interpretation/v1/provenance_manifest.json (historical source; not bundled) | `9a7744bae230928bde433ee0cc4064a9e4feaf89b060f5f9de1e78c0c8107c60` |
| artifacts/thesis_interpretation/v1/validation_receipt.json (historical source; not bundled) | `44508b1964a125999c4e1f97c30ce36e15bc849e447d9efbe7d6dfd064544b59` |
| [artifacts/thesis_interpretation/v1/validation_report.md](../../interpretation/thesis/validation_report.md) | `cd53fe60f660612b64047ee4ae597bb22e2dafc5962c8465005e34f9fd71312c` |
| benchmark_v1/protocol/family_aware_joint_runtime_quality_mps_v1.json (historical source; not bundled) | `0019e8b39b3952561a8753f5ea4055a3b98c8471a6f11617ea339df1487f9563` |
| benchmark_v1/protocol/qualified_logical_benchmark.json (historical source; not bundled) | `e8986cb6435037294c31eecce3536c402a0978b54f21bcdd278a35db8aa4ac32` |
| benchmark_v1/protocol/real_qpu_generalization.json (historical source; not bundled) | `005048f7afb28e1f983db416a799f8542007faac88a6285a46b5fbb337776570` |
| benchmark_v1/protocol/rq4_completion.json (historical source; not bundled) | `8fdfbe85f8c10a6ce00ee8d122676a0eada3cb6be4b2cb6dbd235bebcac3a2b1` |
| benchmark_v1/protocol/simulator_predictor_cells_v2.json (historical source; not bundled) | `31ca1821c609ff86a9585a6321227c40d4bc2b47b76a4c0441943730f8a1fb72` |
| benchmark_v1/scripts/run_qualified_logical_benchmark.py (historical source; not bundled) | `58c20f6077c0796ae85b3625b53a4000a1cda84995dfce2538f97b6a974126dd` |
| docs/research_conclusions.md (historical source; not bundled) | `8180db670677e73442fb9535a22f2d3c71e2e6055e57b4728835393cb0b28b1c` |
| docs/research_contract.md (historical source; not bundled) | `7903e008bfd53fd1dbcac6ffa8a983b4fc919dfa4beedee72c76253319aab4ff` |
| experiments/simulator_runtime_v2/README.md (historical source; not bundled) | `7d88c6b0c5d9e4f32e267f664101207b894c41d2814847811aafba204a484dd2` |
| work/repo_finalization/publication/data/simulator/tensor_network/measurement/cutensornet_scalar_common_raw.csv (historical source; not bundled) | `843a28edae4afc90eec83a491fea21519d67d5f4454093d436200ca65d23535d` |
