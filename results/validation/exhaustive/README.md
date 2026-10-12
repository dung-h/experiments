> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Final exhaustive validation

This package checks frozen benchmark evidence and regenerates indexes/metrics from saved artifacts. It does not decide research claims, change protocols, or repair inputs. No experiment, training, refit, inference, simulator/QPU execution, or new timing measurement is performed.

Start with `validation_report.md`, then review `integrity_report.md`, `metric_regeneration_report.md`, `timing_reconciliation.md`, `observation_alignment_report.md`, and `discrepancy_register.md`. Machine-readable outputs are the four `final_*.csv` files. `build_final_exhaustive_validation.py` records the deterministic procedure.

`recomputed_from_saved_*` rows are recalculated from row-level saved predictions/timing and saved labels. `reconciled_*` rows are copied from frozen reader artifacts whose existing validators were replayed; they are not misrepresented as independent row-level recomputations. The evidence index does not decide whether an RQ is answered.
