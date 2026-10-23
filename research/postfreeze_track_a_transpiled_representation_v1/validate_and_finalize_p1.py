#!/usr/bin/env python3
"""Saved-evidence validation and bounded reporting for P1; never trains."""
from __future__ import annotations
import csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/postfreeze_track_a_transpiled_representation_v1'
PANEL=ROOT/'artifacts/real_qpu/qualified_logical_panel/panel.csv'
OUTER=ROOT/'artifacts/real_qpu/qualified_logical_panel/outer_splits.csv'
INNER=ROOT/'artifacts/real_qpu/qualified_logical_panel/inner_splits.csv'
AGG=ART/'p1_representation'/'aggregate'

def rows(p):
    with p.open(newline='',encoding='utf8') as h:return list(csv.DictReader(h))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def get_metric(table,learner,view):return next(x for x in table if x['learner']==learner and x['view']==view)
def get_cmp(table,learner,left,right):return next(x for x in table if x['learner']==learner and x['left_view']==left and x['right_view']==right)

def main():
    panel={r['canonical_observation_id']:r for r in rows(PANEL)};outer={r['canonical_observation_id']:r for r in rows(OUTER)};inner=rows(INNER)
    errors=[]
    if len(panel)!=4515 or len(set(r['group_id'] for r in panel.values()))!=165:errors.append('frozen_cohort_invariant')
    group_folds=defaultdict(set)
    for r in outer.values():group_folds[r['group_id']].add(r['outer_fold'])
    if any(len(v)!=1 for v in group_folds.values()):errors.append('outer_group_leakage')
    for fold in range(5):
        test={cid for cid,r in outer.items() if r['outer_fold']==str(fold)}
        train={cid for cid in panel if cid not in test}
        inner_ids={r['canonical_observation_id'] for r in inner if r['outer_fold']==str(fold)}
        if inner_ids!=train:errors.append('inner_membership_'+str(fold))
        by_inner=defaultdict(set)
        for r in inner:
            if r['outer_fold']==str(fold):by_inner[r['inner_fold']].add(r['group_id'])
        if set(by_inner)!={'0','1','2','3'} or any(by_inner[a]&by_inner[b] for a in by_inner for b in by_inner if a!=b):errors.append('inner_group_leakage_'+str(fold))
    for learner in ('ridge','hgb'):
        pred=rows(ART/'p1_representation'/learner/'outer_predictions.csv');sel=rows(ART/'p1_representation'/learner/'hyperparameter_receipts.csv')
        for view in ('S0','P0','P1','H0','H1'):
            here=[x for x in pred if x['view']==view]
            if len(here)!=4515 or {x['canonical_observation_id'] for x in here}!=set(panel):errors.append(f'coverage_{learner}_{view}')
            if any(x['outer_fold']!=outer[x['canonical_observation_id']]['outer_fold'] or x['target_seconds']!=panel[x['canonical_observation_id']]['target_seconds'] for x in here):errors.append(f'target_or_fold_alignment_{learner}_{view}')
            selections=[x for x in sel if x['view']==view and x['selected']=='1']
            if len(selections)!=5 or {x['outer_fold'] for x in selections}!={'0','1','2','3','4'}:errors.append(f'selection_{learner}_{view}')
    table=rows(AGG/'pooled_metrics.csv');cmp=rows(AGG/'paired_comparisons.csv')
    report=f'''# P1 representation results

## Facts

The frozen Track-A panel supplied 4,515 observations and 165 group-disjoint
identities.  P0 established complete physical-graph and nominal-duration-proxy
coverage, so every S0/P0/P1/H0/H1 comparison uses the identical 4,515 held-out
OOF rows.  Ridge is the preregistered primary learner; HGB is one bounded
secondary robustness learner.  All hyperparameter selection was inside each
outer training partition.

| Learner | View | Group-balanced MAE (s) | Pooled MAE (s) | Raw R² | log1p R² |
|---|---|---:|---:|---:|---:|
'''
    for learner in ('ridge','hgb'):
        for view in ('S0','P0','P1','H0','H1'):
            m=get_metric(table,learner,view);report+=f"| {learner} | {view} | {float(m['group_balanced_mae_seconds']):.6f} | {float(m['pooled_mae_seconds']):.6f} | {float(m['raw_r2']):.6f} | {float(m['log1p_r2']):.6f} |\n"
    report+='''

Negative deltas below favor the left view.  Intervals are 10,000 paired
structural-group bootstrap draws (seed 42), conditional on these saved fitted
models; they do not add retraining uncertainty.

| Learner | Comparison | Δ group-balanced MAE (s) | 95% CI |
|---|---|---:|---|
'''
    for learner in ('ridge','hgb'):
        for left,right,label in (('P0','S0','Transpiled-static − Source-static'),('P1','P0','Timing − Transpiled-static'),('H0','S0','Hybrid − Source-static'),('H0','P0','Hybrid − Transpiled-static'),('H1','H0','Hybrid+timing − Hybrid')):
            c=get_cmp(cmp,learner,left,right);report+=f"| {learner} | {label} | {float(c['point_group_balanced_mae_delta_seconds']):+.6f} | [{float(c['ci95_low']):+.6f}, {float(c['ci95_high']):+.6f}] |\n"
    report+='''

## Interpretation

The primary Ridge comparison does **not** support H1: Transpiled-static alone
is worse than Source-static.  It does show an H3-style improvement when the
saved nominal duration proxy is added to physical static descriptors, and an
H2-style Ridge complementarity result for H0.  Those directions are not
replicated by the bounded HGB robustness analysis: its relevant intervals all
cross zero except H1−H0, which is a hybrid timing effect rather than a
standalone transpiled-static gain.  The source slices are heterogeneous and
the Qonductor slice is adverse for the Ridge hybrid, so neither the Ridge-only
timing nor hybrid result warrants a broad representation claim.

P2 is therefore **not started**.  The preregistered P1 gate requires a
favorable physical/transpiled representation result not explained by a single
source, no severe tail issue, and a feasible bounded graph design.  Direct
Transpiled-static evidence fails that gate; a topology model would be a rescue
search rather than attribution of an established representation benefit.

## Verdict

**A. No benchmark-scale support for transpiled representation.** This is a
bounded negative result for the declared static representation and timing proxy
under the heterogeneous Track-A target contract.  It neither contradicts the
historical 340-row pilot nor says that exact physical representation is
universally unhelpful.

## Limits

- Qonductor has exact submitted physical structure, while Ma–Li/QPack physical
  views are frozen reconstruction/nominal target views.
- The duration feature is a saved nominal single-shot proxy, not historical
  calibration timing; source target boundaries remain heterogeneous.
- 3,945 QPack rows represent only six groups; group balancing addresses row
  multiplicity but cannot create independent circuits.
- This study contains no physical-DAG message passing and no QPU coupling graph.
'''
    (ART/'p1_representation'/'REPORT.md').write_text(report,encoding='utf8')
    final=f'''# Track-A Transpiled Representation Validation v1

## FACTS

P0 joined all 4,515 frozen observations to saved physical graphs (346 unique
payloads, 374 graph-plus-snapshot keys) and saved nominal duration proxies.
Full-operation physical GNNs are infeasible for this population without
coarsening: 300/346 graphs exceed 50k nodes and 76 exceed 300k.  P1 completed
the predeclared Ridge and secondary HGB comparisons on identical OOF IDs.

## INTERPRETATION

Static transpiled descriptors did not establish an incremental benchmark-scale
benefit over the frozen logical/source representation.  Some Ridge hybrid and
timing-proxy contrasts are favorable, but they do not reproduce as a robust
transpiled-static effect in HGB and are source-heterogeneous.

## DECISION

Do not run physical-DAG P2.  Its precondition was not met.  The existing
logical-DAG control study is reconciled separately in
`logical_topology_existing_evidence/REPORT.md`; it remains a logical, not
physical-DAG, result.

## LIMITATIONS

All timing features are nominal snapshot proxies; source runtime contracts and
physical structure provenance differ.  No claim concerns actual scheduling,
historical calibrations, or QPU coupling topology.

## VERDICT

**A. No benchmark-scale support for transpiled representation.**
'''
    (ART/'FINAL_REPORT.md').write_text(final,encoding='utf8')
    receipt={'status':'PASS' if not errors else 'FAIL','errors':errors,'rows':4515,'groups':165,'learners':['ridge','hgb'],'views':['S0','P0','P1','H0','H1'],'p1_manifest_sha256':sha(ART/'p1_manifest.json'),'aggregate_receipt_sha256':sha(AGG/'receipt.json')}
    (ART/'p1_representation'/'validation_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt,indent=2,sort_keys=True))
    if errors:raise SystemExit(2)
if __name__=='__main__':main()
