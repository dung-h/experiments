# Track-A Transpiled Representation Validation v1 — protocol

Status: P0 coverage/provenance audit only.  This post-freeze study is isolated
from Scientific Freeze v2 and does not replace any frozen result.

## Locked scientific questions

- **H1:** Under the same learner, grouped folds, target, and shared job context,
  `Transpiled-static` improves group-balanced MAE over `Source-static`.
- **H2:** `Source-static + Transpiled-static` improves over both constituent
  views, rather than merely repeating them.
- **H3:** Adding recorded/proxy timing descriptors improves `Transpiled-static`.
- **H4 (conditional P2 only):** Holding binned physical-DAG descriptors,
  static features, model capacity, and training fixed, true dependency
  adjacency improves over node-only/no-edge descriptors.  H1–H3 are not
  evidence for H4.

## P0 scope and provenance rules

P0 reads the qualified frozen 4,515-observation panel and the existing pinned
compiled-unified representation ledger.  It does not compile, transpile,
simulate, train, score a learner, or inspect target error.

Structure provenance is recorded separately from timing provenance.  In
particular, an exact submitted physical QASM does not make a nominal/current
snapshot an historical timing record.  Missing values remain explicit; they
are never changed to zero.

The P0 join must verify: exactly 4,515 canonical observation IDs, 165 groups,
and source counts 340 Ma–Li / 230 Qonductor / 3,945 QPack.  It must only read
the representation graph saved under its existing immutable graph key.

## Planned P1, not yet authorized by this P0 pass

If P0 establishes a defensible common-success population, P1 will use five
frozen group-disjoint outer folds and four inner group folds.  The primary
learner will be log1p-target Ridge with a fixed alpha grid
`[1e-4,1e-3,1e-2,1e-1,1,10,100,1000]`; one equal-budget HGB analysis will be
secondary.  All views receive `log1p(shots)` and no source/backend ID in the
primary model.  P1 views are Source-static (S0), Transpiled-static (P0),
Transpiled-static+timing (P1), Source+Transpiled (H0), and Source+Transpiled+
timing (H1).  Counts/depth/duration use log1p and fold preprocessing is fit
only on outer training data.

P1 primary comparisons are paired group-balanced MAE, with 10,000 structural
group bootstrap draws and seed 42.  Direct contrasts H0–S0 and H0–P0 test H2;
P1–P0 tests H3.  A common-success intersection will be used whenever coverage
differs.  No feature, gate vocabulary, or threshold may be selected from
outer-test performance.

P2 is conditional on a saved P1 report and gate.  It will not reuse the
already-completed logical-DAG topology experiment as if it were a
physical/transpiled-DAG experiment.
