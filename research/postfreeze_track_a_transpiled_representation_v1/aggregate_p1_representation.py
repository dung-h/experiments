#!/usr/bin/env python3
"""Aggregate frozen P1 prediction files; no model fitting occurs here."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/postfreeze_track_a_transpiled_representation_v1"
PRED_ROOT = ART / "p1_representation"
OUT = PRED_ROOT / "aggregate"
VIEWS = ("S0", "P0", "P1", "H0", "H1")
CONTRASTS = (("P0", "S0", "H1_transpiled_static_minus_source_static"), ("P1", "P0", "H3_timing_minus_transpiled_static"),
             ("H0", "S0", "H2_hybrid_minus_source_static"), ("H0", "P0", "H2_hybrid_minus_transpiled_static"),
             ("H1", "H0", "hybrid_timing_minus_hybrid_static"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for b in iter(lambda: handle.read(1024*1024), b""): h.update(b)
    return h.hexdigest()


def write(path: Path, records: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields,extrasaction="raise"); w.writeheader();w.writerows(records)


def metrics(records: list[dict[str, str]]) -> dict[str, Any]:
    y=np.array([float(x['target_seconds']) for x in records]);p=np.array([float(x['prediction_seconds']) for x in records]);e=p-y;a=np.abs(e)
    g=defaultdict(list);s=defaultdict(list)
    for x,error in zip(records,a):g[x['group_id']].append(error);s[x['source_id']].append(error)
    yl=np.log1p(y);pl=np.log1p(np.maximum(p,-.999999999))
    return {'rows':len(records),'groups':len(g),'pooled_mae_seconds':float(a.mean()),'group_balanced_mae_seconds':float(np.mean([np.mean(v) for v in g.values()])),
            'source_balanced_mae_seconds':float(np.mean([np.mean(v) for v in s.values()])), 'medae_seconds':float(np.median(a)), 'rmse_seconds':float(np.sqrt(np.mean(e**2))),
            'raw_r2':float(1-np.sum(e**2)/np.sum((y-y.mean())**2)),'log1p_r2':float(1-np.sum((pl-yl)**2)/np.sum((yl-yl.mean())**2)),
            'p90_absolute_error_seconds':float(np.quantile(a,.9)),'p95_absolute_error_seconds':float(np.quantile(a,.95)),'p99_absolute_error_seconds':float(np.quantile(a,.99)),
            'max_absolute_error_seconds':float(a.max()),'mean_signed_error_seconds':float(e.mean()),'negative_prediction_count':int((p<0).sum())}


def group_errors(records: list[dict[str,str]]) -> dict[str,float]:
    groups=defaultdict(list)
    for x in records: groups[x['group_id']].append(abs(float(x['prediction_seconds'])-float(x['target_seconds'])))
    return {group:float(np.mean(v)) for group,v in groups.items()}


def main() -> int:
    all_records: dict[tuple[str,str],list[dict[str,str]]] = {}
    for learner in ('ridge','hgb'):
        receipt=PRED_ROOT/learner/'receipt.json'
        if not receipt.is_file() or json.loads(receipt.read_text()).get('status')!='COMPLETE_UNAGGREGATED':raise RuntimeError(f'incomplete:{learner}')
        records=rows(PRED_ROOT/learner/'outer_predictions.csv')
        for view in VIEWS:
            selected=[x for x in records if x['view']==view]
            if len(selected)!=4515 or len({x['canonical_observation_id'] for x in selected})!=4515:raise RuntimeError(f'coverage:{learner}:{view}')
            all_records[(learner,view)]=selected

    result=[];source=[];fold=[];tail=[]
    for (learner,view),records in all_records.items():
        result.append({'learner':learner,'view':view,**metrics(records)})
        for field,dest in (('source_id',source),('outer_fold',fold)):
            for value in sorted({x[field] for x in records}):dest.append({'learner':learner,'view':view,field:value,**metrics([x for x in records if x[field]==value])})
        for x in records:
            tail.append({'learner':learner,'view':view,**x,'absolute_error_seconds':abs(float(x['prediction_seconds'])-float(x['target_seconds'])),'signed_error_seconds':float(x['prediction_seconds'])-float(x['target_seconds'])})

    rng=np.random.default_rng(42); group_ids=np.array(sorted({x['group_id'] for x in all_records[('ridge','S0')]}));draws=rng.integers(0,len(group_ids),size=(10000,len(group_ids)),endpoint=False)
    comparisons=[]; distributions={}
    for learner in ('ridge','hgb'):
        for left,right,label in CONTRASTS:
            a,b=group_errors(all_records[(learner,left)]),group_errors(all_records[(learner,right)])
            delta=np.array([a[g]-b[g] for g in group_ids])
            bootstrap=delta[draws].mean(axis=1);key=f'{learner}|{left}-{right}'
            distributions[key]=bootstrap
            comparisons.append({'learner':learner,'contrast':label,'left_view':left,'right_view':right,
                'point_group_balanced_mae_delta_seconds':float(delta.mean()),'ci95_low':float(np.quantile(bootstrap,.025)),'ci95_high':float(np.quantile(bootstrap,.975)),
                'empirical_bootstrap_improvement_fraction':float(np.mean(bootstrap<0)),'bootstrap_replicates':10000,'bootstrap_seed':42,'resampling_unit':'structural_group','sign_convention':'negative_favors_left'})
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'pooled_metrics.csv',result,list(result[0]))
    write(OUT/'source_metrics.csv',source,list(source[0]));write(OUT/'fold_metrics.csv',fold,list(fold[0]));write(OUT/'tail_errors.csv',tail,list(tail[0]));write(OUT/'paired_comparisons.csv',comparisons,list(comparisons[0]))
    np.savez_compressed(OUT/'bootstrap_distributions.npz',group_ids=group_ids,draw_indices=draws,**distributions)
    provenance_path = ART / 'features' / 'representation_manifest.csv'
    provenance={x['canonical_observation_id']:x for x in rows(provenance_path)}
    prov=[]
    for (learner,view),records in all_records.items():
        for struct,timing in sorted({(provenance[x['canonical_observation_id']]['structure_provenance'],provenance[x['canonical_observation_id']]['timing_provenance']) for x in records}):
            sel=[x for x in records if provenance[x['canonical_observation_id']]['structure_provenance']==struct and provenance[x['canonical_observation_id']]['timing_provenance']==timing]
            prov.append({'learner':learner,'view':view,'structure_provenance':struct,'timing_provenance':timing,**metrics(sel)})
    write(OUT/'provenance_metrics.csv',prov,list(prov[0]))
    receipt={'status':'PASS','views':list(VIEWS),'learners':['ridge','hgb'],'predictions_per_view':4515,'bootstrap':{'draws':10000,'seed':42,'unit':'structural_group'},'input_hashes':{str((PRED_ROOT/l/'outer_predictions.csv').relative_to(ROOT)):sha(PRED_ROOT/l/'outer_predictions.csv') for l in ('ridge','hgb')}}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt,indent=2,sort_keys=True))

if __name__=='__main__':main()
