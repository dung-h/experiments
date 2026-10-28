#!/usr/bin/env python3
"""Verify frozen cohort and all reusable baseline neural cells, read-only."""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/postfreeze_mali_shots_ablation_v1'
PANEL=ROOT/'artifacts/real_qpu/qualified_logical_panel'
BASE=PANEL/'neural'
SEEDS=(42,1234,31415)
METHODS=('mali_logical_graph','mali_global_mlp')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with Path(p).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def idhash(ids):return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
def main():
    import importlib.util
    s=importlib.util.spec_from_file_location('qualified_runner',ROOT/'benchmark_v1/scripts/run_qualified_logical_benchmark.py')
    m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
    ctx=m.load_context()
    assert len(ctx['panel'])==4515 and len(ctx['ledger'])==4515
    assert len({r['group_id'] for r in ctx['panel']})==165
    source={s:sum(r['source_id']==s for r in ctx['panel']) for s in ['mali_real_qpu','qonductor_single_circuit_ibm','qpack_mcp']}
    assert source=={'mali_real_qpu':340,'qonductor_single_circuit_ibm':230,'qpack_mcp':3945}
    targets=np.array(list(ctx['targets'].values()));assert np.isfinite(targets).all() and (targets>0).all()
    shots=np.array([int(float(r['shots'])) for r in ctx['panel']]);assert (shots>0).all()
    outer=ctx['outer'];groups={i:r['group_id'] for i,r in ctx['ledger'].items()}
    fold_info={};baseline=[];cell_rows=[]
    for fold in range(5):
        parts=m.partitions(ctx,fold)
        assert len(parts['fit'])+len(parts['validation'])+len(parts['test'])==4515
        assert not ({groups[i] for i in parts['fit']} & {groups[i] for i in parts['validation']+parts['test']})
        assert not ({groups[i] for i in parts['validation']} & {groups[i] for i in parts['test']})
        fold_info[fold]={k:idhash(parts[k]) for k in ('fit','validation','test')}
        testset=set(parts['test'])
        for method in METHODS:
            method_seed_preds=[]
            for seed in SEEDS:
                d=BASE/f'fold_{fold}'/method/f'seed_{seed}'
                r=json.loads((d/'manifest.json').read_text())
                assert r['status']=='complete' and r['epochs']==500 and r['test_rows']==len(testset)
                assert r['identity']['fold']==fold and r['identity']['seed']==seed and r['identity']['method']==method
                assert r['identity']['fit_ids']==fold_info[fold]['fit']
                assert r['identity']['validation_ids']==fold_info[fold]['validation']
                assert r['identity']['test_ids']==fold_info[fold]['test']
                for name,h in r['outputs'].items():assert sha(d/name)==h,(fold,method,seed,name)
                ps=rows(d/'predictions.csv');assert len(ps)==len(testset)
                assert {x['canonical_observation_id'] for x in ps}==testset
                assert len({x['canonical_observation_id'] for x in ps})==len(ps)
                assert all(x['method_id']==method and int(x['outer_fold'])==fold and int(x['seed'])==seed for x in ps)
                assert all(np.isfinite(float(x['predicted_seconds'])) for x in ps)
                method_seed_preds.append({x['canonical_observation_id']:float(x['predicted_seconds']) for x in ps})
                cell_rows.append(dict(fold=fold,method=method,seed=seed,rows=len(ps),best_epoch=r['best_epoch'],
                    elapsed_seconds=r['elapsed_seconds'],manifest_sha256=sha(d/'manifest.json'),
                    checkpoint_sha256=sha(d/'checkpoint.pt'),predictions_sha256=sha(d/'predictions.csv')))
            for cid in testset:
                baseline.append(dict(canonical_observation_id=cid,fold=fold,method=method,
                    prediction=float(np.median([x[cid] for x in method_seed_preds]))))
    # Verify seed-median baseline against canonical saved OOF predictions.
    saved=m.rows(PANEL/'summary/oof_predictions.csv')
    by={(x['canonical_observation_id'],x['method_id']):float(x['predicted_seconds']) for x in saved if x['method_id'] in METHODS and x['status']=='predicted'}
    diffs=[]
    for x in baseline:
        y=by[(x['canonical_observation_id'],x['method'])]
        diffs.append(abs(x['prediction']-y))
    assert max(diffs)<1e-6, max(diffs)
    # Verify panel shots agree with canonical and native rows; never trust compiled cached shots.
    canon={x['canonical_row_id']:x for x in m.rows(ROOT/'artifacts/benchmark_v1/canonical_corpus_8767_20260926/canonical_observations.csv')}
    native={x['canonical_row_id']:x for x in m.rows(ROOT/'artifacts/real_qpu/qonductor_native_features/feature_rows.csv')}
    panel_check=[]
    for r in ctx['panel']:
        i=r['canonical_observation_id'];s=int(float(r['shots']))
        assert s==int(float(canon[i]['shots']))==int(float(native[i]['shots']))
        panel_check.append(dict(canonical_observation_id=i,shots=s))
    receipt=dict(status='PASS',cohort_rows=4515,groups=165,source_counts=source,folds=5,
        baseline_cells=len(cell_rows),baseline_rows=len(baseline),baseline_oof_max_abs_diff_seconds=max(diffs),
        baseline_cell_checks=cell_rows,fold_membership_hashes=fold_info,shots_join='panel=canonical=native all 4515',
        no_model_training=True,no_predictions_recomputed=True,input_hashes={str(p.relative_to(ROOT)):sha(p) for p in [PANEL/'manifest.json',PANEL/'panel.csv',PANEL/'outer_splits.csv',PANEL/'inner_splits.csv',PANEL/'summary/oof_predictions.csv',ROOT/'benchmark_v1/protocol/qualified_logical_benchmark.json']})
    (OUT/'baseline_reuse_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with (OUT/'baseline_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['canonical_observation_id','fold','method','prediction']);w.writeheader();w.writerows(baseline)
    print(json.dumps({k:v for k,v in receipt.items() if k!='baseline_cell_checks'},indent=2))
if __name__=='__main__':main()
