# Final-completion stop/go table

| Experiment | Scientific gap | Needed for current claim? | Cost | Decision |
|---|---|---:|---:|---|
| A1 10/25/50% sample efficiency | Whether strict simulator initialization helps when QPU labels are scarce | YES | Completed | COMPLETE; `artifacts/postfreeze_track_a_transfer_sample_efficiency_v1/aggregation_v2/` validated 360/360 cells. MLP benefit is supported only at 10%/25% fitted-group fractions; Graph benefit and 50%/100% benefit are not established. |
| Classical prediction cost | Complete accuracy–cost frontier | Only if a complete frontier is claimed | Medium | DO NOT RUN under the current timing harness; it lacks the same raw-input/measurement path for fitted RF/XGB/Ridge and would not be comparable without a new contract. Preserve the existing Graph-vs-MLP partial claim. |
| Matched logical-topology Track A | Connectivity-specific claim for the logical Ma–Li representation | NO, if wording remains bounded | Medium/high | DO NOT RUN; `artifacts/postclosure_topology_ablation/v1/` already supplies matched T1–T5 controls and the claim matrix downgrades causality. |
| Transpiled representation P1 | Whether compiled structure improves source/static representation at benchmark scale | YES before any physical-DAG P2 | Medium CPU | COMPLETE; `artifacts/postfreeze_track_a_transpiled_representation_v1/p1_representation/`. No robust representation-wide advantage was established. |
| Physical dependency-DAG P2 | Incremental adjacency value after transpiled P1 | Only if P1 gate passes | High | DO NOT RUN; the pre-registered P1 gate did not pass. Do not use a larger GNN as a rescue. |
| OOD support-distance diagnostic | Explain existing stress-test failures | Useful but not required for current claims | Low | READ-ONLY AUDIT; no model rerun or causal interpretation. |
| Reverse size holdout | Broad size generalization | NO for narrow small-to-large wording | Medium | DO NOT RUN; retain the narrow QPack wording. |
| Crossed backend+structure holdout | Pure hardware generalization | NO; data-limited | High | DO NOT RUN; current backend transfer intentionally permits structural overlap. |
| MPS threshold frontier | Runtime–quality safety claim | Conditional on making safe selector central | Medium | DO NOT RUN; existing selector/quality evidence is explicitly partial and belongs to a different contract. |
| New random models / HPO / architecture search | None | NO | High | DO NOT RUN. |
| Extending Maestro solely for coverage | None after canonical QCSim recovery | NO | High | DO NOT RUN. |

