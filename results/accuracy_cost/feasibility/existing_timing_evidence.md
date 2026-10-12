> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis](../../../docs/thesis/README.md).

# Reusable saved timing evidence

No new latency measurements. Values recalculated deterministically from saved
receipts/telemetry by the read-only closure calculator, with source hashes pinned.

## Thirty QPU neural training-loop receipts

| Method | Saved cells | Min / median / max s | Sum cell s |
|---|---:|---|---:|
| mali_logical_graph | 15 | 972.532014 / 1008.479492 / 1046.694739 | 15143.611649 |
| mali_global_mlp | 15 | 74.197207 / 76.722130 / 83.727960 | 1173.300459 |

Producer: neural_cell (historical source; not bundled).
Timer starts after initial model/optimizer/cache setup, includes training,
validation/cache access and earlier checkpoint writes, then is frozen **before**
final checkpoint write and outer-test inference. It excludes some initialization
and is not pure optimizer kernel time. Sum is sum of cell timers, not wall time
of an entire campaign. Different best epochs do not change the saved500-epoch
loop limit. Descriptive historical cost only; cannot use it as B1/B2/B3 or
causal graph-versus-MLP training-efficiency evidence.

## cuTensorNet component telemetry

Source: raw telemetry (historical source; not bundled);
producer native runner (historical source; not bundled).
3,672 rows = 612 first + 3,060 warm; 603 successful sessions × (1 first + 5 warm)
= 3,618 successful rows; nine failed/timeout sessions yield 54 timeout rows.
Successful sessions cover 201 panel members and 188 unique hashes, each with
three session IDs. Session key is `(panel_member_id,session_id)`, not session_id
alone. Session observations of the same circuit are not independent circuits.

| Saved phase | Independent plan sessions | Median s | Range s | Meaning |
|---|---:|---:|---|---|
| t_network_build_s | 603 | 0.294770156993764 | 0.284819982000045–0.387162211001851 | host converter + operands + Network |
| t_path_search_s | 603 | 0.0125105350016383 | 0.000800148998678196–58.0074547690019 | optimizer/path including hidden costs; not isolated EST |
| t_first_contract_s | 603 | 0.0016592960357666 | 0.00148310399055481–0.0498368949890137 | CUDA-event contraction, not estimator latency |
| t_warm_contract_s | 603 | 0.000137600004673004 | 3.29280011355877e-05–0.0399545288085938 | median of five warm samples per session, then across sessions |

Build ~294.770 ms; path ~12.511 ms; warm contraction ~0.137600 ms.
`t_end_to_end_s` has **0** nonempty rows. Do not sum these medians into E2E,
derive API latency from CUDA-event contraction, or call `runtime_est_s` a timer.
Native code reads RUNTIME_EST after contract_path without a separate timer;
internal cost-model work may be embedded in optimizer stages. Attribute-get
cost and full planning cost are not independently measured here.

## Other receipts and classifications

Dense classical `fit_wall_seconds` surrounds the multi-method fit/predict loop,
attempt assembly and prediction CSV write. It is not per-estimator fit or
inference latency. Reuse only as labeled historical bundle elapsed, not AM1.
Simulator first/warm execution tables and analog stage times measure simulated
work, not prediction overhead. Generalization refit receipts are execution
provenance and must not be pooled into matched parent inference timings.

Directly measured: the stage/loop intervals above. Derived: sums of loop
intervals and per-session-then-across-session medians (hierarchy specified).
Inferred: likely initialization/planning dominance is a hypothesis, not a new
measurement. Missing: B1 inference, full B2 online pipeline, B3 cold setup,
matched peak memory, RUNTIME_EST-internal computation latency and amortization
under a real request stream. No existing evidence closes those AM1 questions.
