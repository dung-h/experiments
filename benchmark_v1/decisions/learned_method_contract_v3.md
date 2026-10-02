# S68 — Learned-method contract V3

**Decision date:** 2026-09-29  
**Status:** signed; no V3 training has started  
**Scope:** source-native learned results and the two new unified adaptations.

## Two reporting classes, never conflated

| Class | Method/panel | Rows | Meaning |
| --- | --- | ---: | --- |
| source-native | C122 Qonductor | 4,482 | grouped source-local retrain/audit on exact submitted physical QASM |
| source-native | C123 Ma--Li | 340 | `mali_source_local_340`; exact logical-QASM source context |
| unified adaptation | Qonductor polynomial V3 | 8,767 attempted rows | one shared outer split; source-agnostic structural feature contract |
| unified adaptation | Ma--Li graph architecture V3 | 8,767 attempted rows | one shared outer split; graph-architecture adaptation, not paper reproduction |

“Attempted” means every canonical row gets a method-row envelope record.  It
does not mean every row is guaranteed a valid pre-execution representation.
Failures stay in coverage denominators; numerical MAE pairing uses only the
identical successful outer-test IDs.

## Common split and firewalls

Both unified V3 methods use exactly the V2 frozen outer assignment and its
grouped inner-fold construction.  For every outer fold, preprocessing,
feature construction vocabulary, degree selection, model selection, early
stopping and target transformation are fitted solely inside outer training.
No method may create a source-specific split, tune against outer test error or
select a preferred seed.

The target is `log1p(target_seconds)` for fitting and `expm1` for reported
seconds predictions.  All randomized graph runs use seeds `42`, `1234`, and
`31415`; retain every seed output and use the rowwise median as the declared
reduction.  The graph runner must use CUDA and record a CUDA smoke test,
environment lock, device, driver, precision and memory.  Polynomial fitting
is deterministic CPU work under a fixed thread cap.

## Primary unified polynomial V3

This is a Qonductor-style **adaptation**, not an assertion that Qonductor's
published coefficient semantics are identical across sources.  Its complete
input feature vector is:

```text
log1p(active_width)
log1p(structural_depth)
log1p(two_qubit_count)
log1p(swap_like_count)
log1p(shots)
circuit_count = 1
```

It uses `PolynomialFeatures + LinearRegression`, selecting degree 2, 3 or 4
by source-balanced inner-fold MAE after inversion; lower degree breaks exact
ties.  It must not use backend one-hot, backend code, source ID,
representation tier, source-specific missingness mask, workflow ID, circuit
hash, dataframe order, a target-derived statistic, or post-execution
metadata.  A future backend-context variant is a distinct diagnostic method
ID, not an ablation silently merged into V3.

## Unified Ma--Li graph architecture V3

This is a new graph-architecture adaptation, not literal Ma--Li replication.
It consumes the V3 DAG representation, directed edges, gate type/arity,
parameter-presence and qubit-position availability plus the source-agnostic
global structural quantities above.  No source ID, backend one-hot,
representation-tier code, circuit hash, workflow ID or missingness flag enters
the primary model.  QPack numerical angles are never fabricated.

Gate vocabulary/unknown token, scalar normalizers and any early-stopping
decision are train-fold only.  It uses the declared public directed-DAG
message-passing/mean-pooling architecture but initializes from scratch; no
source-native checkpoint is reused.  The first execution is fold 0 at all
three seeds and must pass feature-firewall, split, CUDA and OOF-ID QA before
folds 1--4 begin.

## Supersession

This contract supersedes
`execution/manifests/rqpu_unified_adaptations_v2.json` for the two unified
learned adaptations only.  V2 remains preserved as an invalid/quarantined
scientific checkpoint, not a source of V3 predictions.  C122 and C123 are not
retroactively changed: C122 awaits the S67 audit; C123 is relabelled
`mali_source_local_340` and is not a unified 8,767 result.
