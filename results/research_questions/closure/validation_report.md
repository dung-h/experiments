> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Closure validation

```json
{
  "status": "PASS",
  "task": "reporting and scientific closure only",
  "readme_checks": 6,
  "readme_actual_namespace_matches": 6,
  "pathless_declarations": 182,
  "sha256_declarations": 180,
  "git_revision_declarations": 2,
  "unresolved_declarations": 0,
  "primary_claim_provenance_risks": 0,
  "rq_track_rows": 10,
  "claims": 16,
  "blocking_scientific_gaps": 0,
  "required_new_experiments": 0,
  "source_files_checked": 110,
  "source_hashes_unchanged": true,
  "deterministic_saved_evidence_validators": [
    {
      "command": "/home/server/Documents/.venv-qonductor/bin/python -B artifacts/final_research/verify.py",
      "exit_code": 0,
      "status": "PASS",
      "scope": "existing deterministic saved-evidence validator; scientific closure assessed separately"
    },
    {
      "command": "/home/server/Documents/.venv-qonductor/bin/python -B benchmark_v1/scripts/validate_benchmark_interpretation.py",
      "exit_code": 0,
      "status": "PASS",
      "scope": "existing deterministic saved-evidence validator; scientific closure assessed separately"
    },
    {
      "command": "/home/server/Documents/.venv-qonductor/bin/python -B benchmark_v1/scripts/report_am1_phase4.py --stage validate",
      "exit_code": 0,
      "status": "PASS",
      "scope": "existing deterministic saved-evidence validator; scientific closure assessed separately"
    },
    {
      "command": "/home/server/Documents/.venv-qonductor/bin/python -B benchmark_v1/scripts/validate_am1_phase5_dense_b2.py",
      "exit_code": 0,
      "status": "PASS",
      "scope": "existing deterministic saved-evidence validator; scientific closure assessed separately"
    }
  ],
  "original_exhaustive_package_preserved": true,
  "scientific_freeze_preserved": true,
  "new_training": false,
  "new_inference": false,
  "new_workloads": false,
  "new_timing": false,
  "historical_manifests_modified": false,
  "commit_push": false
}
```

All ten RQ/track rows have one allowed status and explicit blocking decision. Claims contain verified local source paths/SHA-256 and saved metric records. P4 exact45, P5 exact30 and Dense full150 remain separate; source-balanced, pooled and equal-group errors retain distinct fields. Seconds-to-milliseconds conversion is used only for the displayed Dense table. Conditional CI signs are checked for Dense pair statements; QPU paired interval includes zero. No source statistic is invented or reselected.

Working RQ mapping is stated without rewriting Research Contract v1.1. Adaptation/native/proxy labels, QNN/Random failures, collisions, QCEC zero-input boundaries, subset-selection and cross-environment timing limits remain visible. Earlier v1 status is preserved and superseded only for current AM1 closure. All source hashes captured by this builder are checked again before writing.

`--check` verifies every derived payload byte and rechecks source identities, mappings and source evidence. `git diff --check` is executed after writing/checking; its output is returned by the command. Existing exhaustive test/validator results are referenced from the immutable prior package; no new full-suite PASS is claimed here.

## Captured source hashes

| path | sha256 |
|---|---|
| /home/server/Documents/qcre/estimate.py | e85873a810519c2741df281de5f9a8c6a89cedf80c17da65b74c7df39fa497c0 |
| AGENTS.md | 8ffe896fafd5a2881b41d6590c6c481a7a7b5afd8d8f3d31bcebe475e05b666b |
| README.md | 377560704622734ed6c3b81228163948debbd7e2dadb04564d13daf47e8526e9 |
| artifacts/am1_feasibility/v1/README.md | 47bb071038830e10820fcd9eb96ac60fa947e6cd973585d9be6f67c5c375e686 |
| artifacts/am1_feasibility/v1/provenance_manifest.json | 95255767b5c0e992f1e53ad13bf53e44cf13b4d446e222a09a7e795c7e2c37c5 |
| artifacts/am1_overhead_pilot/v1/README.md | 712a4e2ec6677e716f1fd34fd987602c1b123eee787899cad15a3702fa48764b |
| artifacts/am1_overhead_pilot/v1/provenance_manifest.json | 0c09e8af3b456b4bdf429d0eb544124b98959f867426a3c2b9cdbd1f454d1700 |
| artifacts/am1_phase2/v1/README.md | 49d9ce9503b42187274c5a42521ae03a7e52bc1aade8f066226225239424d0f5 |
| artifacts/am1_phase2/v1/provenance_manifest.json | fa45cd39a668add920b4d252e86fb4d56cb69163702a80817496dc704595a4c0 |
| artifacts/am1_phase4/v1/accuracy_cost_tradeoff.md | b20ddfc3a2e433522e383cd49aec46600c4fd5363ef1a2b50b038f38013eea0a |
| artifacts/am1_phase4/v1/analysis.json | 786bf48bce939d8d002d01313004e7309e7abd85515b874a92c09ccfdc71ca1a |
| artifacts/am1_phase5_dense_b2/v1/analysis.json | 89e0ebaeb4b2111a52bba39bfd15aa618bb7243d4a3d017cc51476909eed4d26 |
| artifacts/am1_phase5_dense_b2/v1/full_context_accuracy.csv | 8fd52f30222140eb91ed9870e4f35efd14a1d149fd7167d75625cf3efca0c2fc |
| artifacts/am1_postpilot_audit/v1/README.md | e5b8d3907392a2edecbed5627577a30ec3b1996120a63f9909509733686569e7 |
| artifacts/am1_postpilot_audit/v1/provenance_manifest.json | 5758fc4c3d7d80e9d66245d07fc8c7caefac8b8ac3d1bfbfcccb4c213450e6a7 |
| artifacts/am1_preprocessing_recovery/v1/historical_environment_provenance.md | 8d4708cda09dc42021d4b5d710f5896838c13eae630e6ecec35b3343bf5793c1 |
| artifacts/am1_preprocessing_recovery/v1/root_cause_analysis.md | 4cd5b2dbd3edc28ab45c580bb2bd70f6861c90069ac03bf88517d53231cd5227 |
| artifacts/benchmark_deep_dive/tables/analytical_controls_v1.csv | f293c98475b01f15a38fcf83a17bed48165a6056b63047d231b1f20c0b247f74 |
| artifacts/benchmark_deep_dive/tables/mps_selection_safety_v1.csv | d564edf1557fbb3400bf192c6805639e9b06ee70ddc7aa76639c606496772f47 |
| artifacts/benchmark_deep_dive/tables/pasqal_stage_quality_v1.csv | 9a9ee910b40fd509873c2f997d7b6166402f36f42a48e4f43209305e448c8409 |
| artifacts/benchmark_deep_dive/tables/qpu_metrics_v1.csv | 958552fa0d376e7c6609cfd85f51ce3d147fd74ffb239a3054db9f432659659f |
| artifacts/benchmark_deep_dive/tables/qpu_source_metrics_v1.csv | dac5741fcef81c6f59d40b39abe3d65306b82a680080c8b7de13d5d16b415fb6 |
| artifacts/benchmark_deep_dive/tables/simulator_metrics_v1.csv | b7f11383bccf7022f4b26b207cd3c95a1b830676158223d6058d2a8979fd3535 |
| artifacts/benchmark_deep_dive/tables/simulator_paired_v1.csv | 6958ebf05498a6aed739b10f8dff3a074a81e9bc160b226f8557d0f7ef6893cb |
| artifacts/benchmark_deep_dive/tables/tn_same_plan_slices_v1.csv | 0e64f0f0fee78ecc38e593c60d94c1610cff35c78bd440856ce3bcb58024f4dd |
| artifacts/benchmark_interpretation/v1/mali_feature_schema_audit.md | cb354db82584b2b8f8b4b30b87c596b7cca61eac91de9a22d312974501efd47c |
| artifacts/final_exhaustive_validation/v1/README.md | eb99dacd62d59c7e66ebc6c904d9329b228efabfc6c921aeedb7d9975b7f3915 |
| artifacts/final_exhaustive_validation/v1/build_final_exhaustive_validation.py | 4f3c1ec45d326fb0e92a4d05ce3cd2f1a0593dae487652d43cdd02df3171e672 |
| artifacts/final_exhaustive_validation/v1/discrepancy_register.md | b1f6ffe5d53e8f2f98814357424fe60df333a62ee10e80553616a2b23d30d5d2 |
| artifacts/final_exhaustive_validation/v1/final_artifact_manifest.csv | e16dd4f69c573d4f823f8eb45b0c057c5238172c51acfc6eb6f7af17d67f35a6 |
| artifacts/final_exhaustive_validation/v1/final_comparison_registry.csv | a3b4268b72c1ec4e1e1a5eac1c2e57f57c33cd8f6a1fdefc2c35fa3cb8c051f7 |
| artifacts/final_exhaustive_validation/v1/final_metric_registry.csv | 70dae146cde332af0e867646b1d5922eb9443a71c0d7a7e6d8b0e586c1f4aee1 |
| artifacts/final_exhaustive_validation/v1/final_rq_evidence_index.csv | ced82c29a4724185e14a4dab986f50d24cb02b8a167c65410a0b795d7c297968 |
| artifacts/final_exhaustive_validation/v1/integrity_report.md | e7753ad5625fda2c6f5e2397137ae547bdac054b48ef38b3cff1c334e03b9252 |
| artifacts/final_exhaustive_validation/v1/metric_regeneration_report.md | abaa3d788a641c5c24f75d328990e16c0aeb5243efbaf8737f21e414b3d39eea |
| artifacts/final_exhaustive_validation/v1/observation_alignment_report.md | ea32b351bee03dbd2b756deadc7f71785b2f27e646855a7d32648927eaa0c268 |
| artifacts/final_exhaustive_validation/v1/test_execution_report.md | 8dd47febf6aeb26d33b1489cf0b6ca00880515fbda425af159c7ffc86a6446fe |
| artifacts/final_exhaustive_validation/v1/timing_reconciliation.md | bcc621ad2cb6072fd32421fb6a09eb2687e2c6bd647d46fc976319b16fb4bd2e |
| artifacts/final_exhaustive_validation/v1/validation_report.md | 7e9f0dfdae5c8ca3f94bab80696eb7c8f08a8246b0418e13edf1d99479f190d1 |
| artifacts/final_research/manifest.json | e94891ac3da7b922f93dc2c1953a2ce38d0c84f28149584db41a6a11e89dab62 |
| artifacts/final_research/rq4_report.md | 0d710cb5da210eac2c30db59663aa998d04b37ecaafc23059d8a6a53f2ce8238 |
| artifacts/final_research/validation.json | 4af7b9f145c814c51af09459e34dfa14505797d567f8281a73ead1bcec171cf0 |
| artifacts/final_research/verify.py | dcc95101a323db4cfdd5e79f58d85ccf97764049f0aba2e983d2d073281723f3 |
| artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011/method_metrics.csv | 559de395c14ee3d990e3054bcf04451e3ec9ad82ff6648d81b57d24da012e801 |
| artifacts/real_qpu/generalization/manifest.json | 51e56821ef9b197af90c0edaf25dfa0f3325c265a08b4788a5eb1a30f1cd8445 |
| artifacts/real_qpu/generalization/splits.json | 7b71dc8e19d1063ecd4701ed3609b6124ee9cea792e7407cf6ac0aefcb19f6a0 |
| artifacts/real_qpu/generalization/support.csv | 94aa8768ae338deecb41200c59705406db1cd0642a11d55f4312f8492ee7de6c |
| artifacts/real_qpu/generalization/targeted_backend_followup_v1/runs/backend_context__ibm_osaka/neural/cells/mali_global_mlp/seed_1234/receipt.json | 9c5c11136bad8cc0cc1d595ede52ac1956b6dd082143948f8dd5b829cee0aaef |
| artifacts/real_qpu/generalization/targeted_backend_followup_v1/runs/backend_context__ibm_osaka/neural/cells/mali_global_mlp/seed_31415/receipt.json | 7da68edd187ec75947a977f926da19aa7f653614510985903479a420fe693771 |
| artifacts/real_qpu/generalization/targeted_backend_followup_v1/runs/backend_context__ibm_osaka/neural/cells/mali_global_mlp/seed_42/receipt.json | ecd67caa46e15c12587fd939f823f943d70c8daa3f0a7591b238a77f63ec4546 |
| artifacts/real_qpu/generalization/targeted_backend_followup_v1/runs/backend_context__ibm_osaka/neural/cells/mali_logical_graph/seed_1234/receipt.json | 65efae2a20bcb1c743b7c829a0a29715b9b7aca9954e79f43d59b72331f49362 |
| artifacts/real_qpu/generalization/targeted_backend_followup_v1/runs/backend_context__ibm_osaka/neural/cells/mali_logical_graph/seed_31415/receipt.json | aa11947f41f5a6a21a9e9b1d5c77b5db932748a1c9bb70d1675b56a9f8bf3e2b |
| artifacts/real_qpu/generalization/targeted_backend_followup_v1/runs/backend_context__ibm_osaka/neural/cells/mali_logical_graph/seed_42/receipt.json | 835dcddd5b08e8bda08181e9ff8c76ddccfd2f6bb4857e3737babc57ef72706d |
| artifacts/real_qpu/generalization/targeted_backend_followup_v1/runs/backend_context__ibm_quito/neural/cells/mali_global_mlp/seed_1234/receipt.json | 8fa6895ef8c231324daa0de2a95dbc1cff51c5cc544c40372fb27a27fb44279d |
| artifacts/real_qpu/generalization/targeted_backend_followup_v1/runs/backend_context__ibm_quito/neural/cells/mali_global_mlp/seed_31415/receipt.json | b5e43b6c130da7fc22aace09ec0e0814a64fdd23ec543757a4856f32a1f584d2 |
| artifacts/real_qpu/generalization/targeted_backend_followup_v1/runs/backend_context__ibm_quito/neural/cells/mali_global_mlp/seed_42/receipt.json | 719b1047d752108442b7ece85a50666c045829323da7fc2a1fbb488bb4d6bd87 |
| artifacts/real_qpu/generalization/targeted_backend_followup_v1/runs/backend_context__ibm_quito/neural/cells/mali_logical_graph/seed_1234/receipt.json | b81561625cd3e419843f237d07b9be74813c06ca2da2dcd374bcbd36c58676b4 |
| artifacts/real_qpu/generalization/targeted_backend_followup_v1/runs/backend_context__ibm_quito/neural/cells/mali_logical_graph/seed_31415/receipt.json | 3749093e35c93d720c70bdb9be2902e64b4ed2a5a38952cab07c3cfc1fcc01d0 |
| artifacts/real_qpu/generalization/targeted_backend_followup_v1/runs/backend_context__ibm_quito/neural/cells/mali_logical_graph/seed_42/receipt.json | 1d6d496d491be6096ad9e396db4b7a48705a72e26d0756ec9b1db8583c7777b7 |
| artifacts/real_qpu/provenance_equivalence/report.md | 04fdbb588c3ad39e3b818cfd2aded6f55562fa218b3763d50f96bcf72753e4de |
| artifacts/real_qpu/qualified_logical_panel/analytical/execution_receipt.json | 3626d951d37c77b260a5d9f2c3e25399315cb04c50e9c3d2549278039cb4a2e7 |
| artifacts/real_qpu/qualified_logical_panel/analytical/manifest.json | dcf5f430a08e0eed8174c0b7ebdb536cd0f8ef097eb1de08d5bc9ff99ce08cbc |
| artifacts/real_qpu/qualified_logical_panel/analytical/qcre_source_receipt.json | 880ceacff2d403323339ab7d2e55a20c9447203eb4308a5cf2475827ab383d37 |
| artifacts/real_qpu/qualified_logical_panel/inner_splits.csv | 057aa6396e772a76a838467c6658b9ee3a89c2f35e9cb307a6ac9ae862b1ffd2 |
| artifacts/real_qpu/qualified_logical_panel/manifest.json | 687c3357cda8523d988641935bd6c5c99d5fed164e85ef6e649bed6e9556d2cb |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_0/validation.json | 0b5f0d6e0453ead54042e45ad802be46eea7a584f1c28a0d01f161e9cb0ddd87 |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_1/validation.json | 47284cfcbaf4e02f3857f239642c139eac379916892963fd92ac7fe347a4cad1 |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_2/validation.json | 4b06bfe7b7ec82943fdf02d3ae1461dc40fada91d20e3e37f0d2357a78e8c698 |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_3/validation.json | bce3ca5c65007498e8f03a3fea8032b439a6e880fb03812ff98eb051d9b3ceed |
| artifacts/real_qpu/qualified_logical_panel/neural/fold_4/validation.json | fc27cc188681db94e1bc7635f6da555a6b00bb24ff52f5f4e596a2133a7fd0db |
| artifacts/real_qpu/qualified_logical_panel/outer_splits.csv | eb80db3c5443498f8f262c3479cb09fb34028c85a26203284118968ff96196c2 |
| artifacts/real_qpu/qualified_logical_panel/panel.csv | b4ade914955d7807e466cf5cf0c5265d21b39a81fdcee8e91dc2fb6300eb978a |
| artifacts/real_qpu/qualified_logical_panel/summary/README.md | d5c1ec365d3c9752bc2c7eb51f28dbfd2b252e69817746333763af6651992e5e |
| artifacts/real_qpu/qualified_logical_panel/summary/final_validation.json | 3c3f50799acaccb83d72f9b7065e116853d7ba7c9d2375ee9984b72894d506ab |
| artifacts/real_qpu/qualified_logical_panel/summary/manifest.json | 42233d628a9c48c5abdb1cfe16a6f10a7b8b1385e923193c1589697916e6c599 |
| artifacts/real_qpu/qualified_logical_panel/summary/oof_predictions.csv | c46634505e9d84f83c65f28c3097c6dbb964ee7d3932e85862d2ed4953ea41bc |
| artifacts/real_qpu/qualified_logical_panel/summary/shared_row_comparisons.csv | f620fd6cf157b8a3cb7491a6d8fe9f116c75e65e60b774101ee95bb360110919 |
| artifacts/real_qpu/strict_logical_panel/analytical/adapter_preflight.json | 32789033ea409e590e9328e00b735b59ab44257c5fca99349031144fdc420859 |
| artifacts/real_qpu/strict_logical_panel/analytical/raw_attempts.csv | 01e646b2bca136370e02c53875c2c0753e515dc33ba033dda2b18c686caab0a7 |
| artifacts/real_qpu/strict_logical_panel/analytical/run_manifest.json | 7336f8f18c32ab7746bf0697801f80c23028f82174c8a83bad6b5405ed515280 |
| artifacts/release_review/scientific_freeze.json | 6253f9943f4fe37851c11eb5b60ac2a779f51891d94a1755dd6a0e9a2b9b4cf3 |
| artifacts/research_question_closure/v1/README.md | f46aa35ea5f61bdfa040c798cd97bc861668c5080891145896fec97914a9e64f |
| artifacts/research_question_closure/v1/provenance_manifest.json | 00462ce8d605dc10ed20df98dee2f7de6147ede7f6aae7734fa1a19190c056a5 |
| artifacts/research_question_closure/v1/rq_framework.md | b5cbad902f4ba68c3cd1b07c151556c76d7549118cb19a6c095db0248e3f6a5a |
| artifacts/thesis_interpretation/v1/estimator_fidelity_registry.md | dd2589afa5117fc734d3773c6718ed2cb0fd34e8591523adae18a647e53d2a67 |
| benchmark_v1/protocol/qualified_logical_benchmark.json | e8986cb6435037294c31eecce3536c402a0978b54f21bcdd278a35db8aa4ac32 |
| benchmark_v1/protocol/real_qpu_generalization.json | 005048f7afb28e1f983db416a799f8542007faac88a6285a46b5fbb337776570 |
| benchmark_v1/registry/method_fidelity_registry_v2.json | 98f3460c8d9a0cbabb424555dd695cee573409ed65facdb60557d836f1750810 |
| benchmark_v1/scripts/aggregate_real_qpu_generalization.py | 674b7e6495b6e80f0f673c3b5be5f75c89b99a9f0971ed3df399fb54da825cd6 |
| benchmark_v1/scripts/analyze_am1_phase2.py | 1368d1e40efa563d289290aeb562ac25cc2f107030fc4d23dc8ce3e377da54a1 |
| benchmark_v1/scripts/build_am1_feasibility.py | 7c373659e558b5924a847715bd7d9d60970901aff79ea71788948cd07b3d53c7 |
| benchmark_v1/scripts/build_am1_postpilot_audit.py | d0c3806421abac621c8b8c064a69d873cd989d45a5d5417db2aedb1e95a78210 |
| benchmark_v1/scripts/build_final_rq_closure_v2.py | 81ad8da20886d2cb16ee34eed142348cd008e9ccb98b729ad66794111e371da9 |
| benchmark_v1/scripts/build_research_question_closure.py | f612247cf12c38d94b7b88ec825cc7ca9dc3f475f154e4b145803c35dfdd445f |
| benchmark_v1/scripts/finalize_qualified_benchmark.py | 95b0fb50cb72a6b346b0c8055544bec8bf8aef303edf4d779ad84b8d8e504987 |
| benchmark_v1/scripts/gpu_lease.py | 06915539e483e06ed8cb76e783eb649103fb06fecd9a69c8d1698b7e59b85421 |
| benchmark_v1/scripts/mali_full_features.py | 7aacb91d57b4efed1cc0c7d5314633bf81fd475a5e4c1b0c904065dde33735d1 |
| benchmark_v1/scripts/report_am1_phase4.py | 1dc43084b0a5ecb117b91788246bceb0a76c4036cafd645c87717e66c4ded1f3 |
| benchmark_v1/scripts/resume_qualified_analytical.py | 8f2826837b53b5cdee117c3f24b00cac848695172912ba54da2725b1b80a093e |
| benchmark_v1/scripts/run_generalization_neural.py | 26291d210b144007fe083aa624d90d0613a66f34555d9638f0a5693c89244640 |
| benchmark_v1/scripts/run_mali_batched.py | 31ecca0d1794fa654281b04a04a70245bf7672be4ecc92ad4593a43df4bbd13a |
| benchmark_v1/scripts/run_mali_full_features.py | 02fd605cb28da81908e10e31e012cc51440541991b2b9cba92f9675fe7678d7e |
| benchmark_v1/scripts/run_qualified_logical_benchmark.py | 58c20f6077c0796ae85b3625b53a4000a1cda84995dfce2538f97b6a974126dd |
| benchmark_v1/scripts/run_strict_logical_analytical.py | d0f27224cdc0332bd24caa350c93ffd182ae89616bb772c74754d1217550690e |
| benchmark_v1/scripts/summarize_am1_overhead_pilot.py | af795b9f6da60f7e4509ffadc5ba5d6469b087e30977307be3ffdc648b40c9ab |
| benchmark_v1/scripts/validate_am1_overhead_pilot.py | 6931ca2d997973deb5f713bf4f0471a3612e4ffb07bf6f010c21b432b4fea499 |
| benchmark_v1/scripts/validate_am1_phase5_dense_b2.py | 9a5a7962a36b3e98a2caf87e4f3c1de77f4b31a26eb67c8d34179c5d783925f0 |
| benchmark_v1/scripts/validate_benchmark_interpretation.py | d64f298562b7ba65f3008c53ba8cae18aa88e0dee6e627792ba7c14ea772694f |
| docs/research_conclusions.md | 8180db670677e73442fb9535a22f2d3c71e2e6055e57b4728835393cb0b28b1c |
| docs/research_contract.md | 7903e008bfd53fd1dbcac6ffa8a983b4fc919dfa4beedee72c76253319aab4ff |
