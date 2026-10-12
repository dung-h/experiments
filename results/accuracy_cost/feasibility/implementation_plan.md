> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# AM1 implementation plan — not a harness

Amendment identifier: **AM1-ESTIMATOR-OVERHEAD**. Proposed future output:
`artifacts/amendments/am1_estimator_overhead/run_001/`. This directory is NOT
initialized by this task. No timing code or adapters have been implemented here.

## Approval and phases

0. Owner approves narrowed method list, B1/B2/B3, workload cap, precision/device,
   budgets, and whether native cuTensorNet planning is included. BLOCKED models
   stay blocked; locate existing exact states only. No training/refitting.
1. Implement a separate inference-only adapter layer, never call runner `main`,
   `--stage all`, evaluators that fit Ridge, or resume methods. Reuse pure
   feature/constructor/formula functions after static side-effect review.
   Parent: `run_qualified_logical_benchmark.make_model`,
   `run_mali_full_features` global/node transform application and pinned model
   constructor, `run_mali_batched.ExampleCache` feature materialization. Separate
   constructor from all fit/cache-to-disk effects; no frozen file changes.
   Dense: `_make_model`, saved `fold_inputs.json`, `_predict` semantics from
   `run_cudaq_dense_neural_predictors_v1.py`. Joint:
   `train_family_aware_joint_mps_v1.static_features` and saved head/classifier
   architecture; do not invoke `prepare_fold` if it fits a classifier.
   Analytical: pinned QCRE `estimate_runtime` + BQSKIT converter; native duration,
   Hyb/proxy functions in analytical producers, saved bridge coefficients.
   Maestro: `load_component_model`/`component_predictions` in pinned evaluator,
   but not nested_ridge_fold/whole evaluate routine. Native TN optional:
   strict_unitary/converter/Network/contract_path/optimizer_value only, never
   child/parent runner (they execute contraction). No historical source edit.
2. Materialize **new** query/adapter manifests from existing inputs only. Preserve
   row IDs, source/physical/recipe/feature/snapshot digests and frozen transforms.
   Hash inputs/outputs before/after. Formal read-only source checks precede
   any approved inference. Fail if source pin or eligible output semantics differs.
3. After separate execution approval, load trusted exact checkpoint states
   under compatible environment, eval/no_grad. Run parity on the selected frozen
   held-out rows before latency: parent best state, dense final state, joint
   runtime+quality+family classification. Expected predictions from pinned
   saved files, not a newly fitted model. For frozen FP32 neural output use
   atol=1e-6 seconds + rtol=1e-5; quality probability atol=1e-6/rtol=1e-5 and
   exact quality decisions. Native scalar/analytical atol=1e-12 seconds +
   rtol=1e-10. These are proposed numerical replay gates, not a license to alter
   model thresholds; any failure requires review, no tolerance widening. Report
   small discrepancies, never overwrite saved predictions. Verify recovered
   reference scalar equals all saved fold/source reference predictions exactly.
4. Run serial isolated B1/B2/B3 using the [protocol](fair_comparison_protocol.md).
   Warm repetitions=30, warmups=10, cold restarts=3, batch1 primary. No automated
   retry, budget expansion or stage fallback. Keep raw timing and system records.
5. Summarize after per-adapter gates. Join frozen accuracy only by identity and
   compatible context, report blocked/failure coverage. Recompute tables from
   raw timing, cross-check stage composition, denominator and uncertainty.
   New timings/calculations never become replacements for frozen predictions.

## Raw and derived schema

Future raw JSONL/CSV fields: amendment/run ID; request ID/canonical observation
ID/QASM hash/rung; track/context/representation/snapshot/feature digest;
structural_group/source/family/width/depth/gates/nodes/edges; fold; method;
model/checkpoint/transform hash; seed list and aggregation; boundary/stage;
cache state; cold process/repetition ID; batch size/member IDs; start_ns,
end_ns, elapsed_ns; host/device timer type; synchronization; setup included;
status/reason; prediction parity max difference; process/child RSS high-water;
Torch allocated/reserved GPU bytes; sampled process GPU bytes; thread/affinity;
software/environment pin; instrumentation-control ID; load/temperature/clocks.
Labels/predictions referenced separately, never written over original artifacts.

Summaries: latency seconds=(end_ns-start_ns)/1e9; per-request median/p90 over
retained valid repetitions; report censoring/failure/timeouts separately. Group
uncertainty resamples independent structural groups, not timing repetitions;
within-query jitter is separate. Batch throughput=batch_n/measured_total_s and
amortized request time=measured_total_s/batch_n, not reciprocal median if
aggregation differs. Peak memory=max observed minus recorded baseline only
when same metric/process lifecycle supports it; also retain absolute peak.
Formula details and comparability rules in the two companion protocol docs.

## Proposed resource budget (unmeasured planning bounds)

- One measurement worker, no parallel training/simulator job. CPU native threads1,
  Torch CPU threads2; CUDA required for neural. Budget host RAM8 GiB, neural
  GPU allocation8 GiB, extra disk<=256 MiB; disk floor3 GiB plus pending write
  reserve. Samples stay small; no checkpoints copied unnecessarily.
- Two-hour hard wall ceiling **for a pilot**, not an ETA or promise of all
  repeats/contexts. At the 50/30 caps and 30 reps, query cost is unknown until
  approved pilot. Request timeout30s, B3 startup120s; stop and request review
  rather than automatically extend. Report any unfinished cell as budget-stop,
  not method failure. Optional fresh TN planning is excluded from this initial
  budget: frozen workspace14,947,909,632 bytes exceeds the8GiB neural budget;
  a separate native-plan budget and authorization are needed, not a silently
  reduced workspace/new estimator. No contraction is ever part of AM1 overhead.
- Resource caps are maximums, not measured required memory; if largest eligible
  graph cannot fit, report censored feasibility, no batch/model change. Use no
  destructive cleanup or data regeneration. Warm asset caching and cold
  filesystem state disclosed; avoid flushing system cache.

Stop on changed scientific input/hash, unknown checkpoint identity, prediction
parity failure, hidden fitting/calibration/simulation, incompatible output clock,
resource floor/OOM, unexpected concurrent GPU owner, or unsupported state-load.
Retain logs/partial results in amendment namespace; do not repair protocols,
drop difficult rows or install a different framework to force success.
