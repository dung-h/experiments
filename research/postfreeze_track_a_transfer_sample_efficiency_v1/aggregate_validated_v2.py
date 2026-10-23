#!/usr/bin/env python3
"""Validate and aggregate saved P1 predictions only; never import training code."""
import csv
import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'artifacts/postfreeze_track_a_transfer_sample_efficiency_v1'
OUT = STUDY / 'aggregation_v2'
PARENT = ROOT / 'artifacts/postfreeze_track_a_transfer_v1'
PANEL = ROOT / 'artifacts/real_qpu/qualified_logical_panel'
SEEDS = (42, 1234, 31415)
TREATMENTS = ('M-S', 'M-P', 'G-S', 'G-P')


def read(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def ids_sha(ids):
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()


def write(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def metrics(rows):
    y = np.array([float(r['target_seconds']) for r in rows])
    p = np.array([float(r['prediction_seconds']) for r in rows])
    e = p - y
    groups, sources = defaultdict(list), defaultdict(list)
    for r, a in zip(rows, abs(e)):
        groups[r['group_id']].append(a)
        sources[r['source_id']].append(a)
    den = np.sum((y-y.mean())**2)
    return dict(rows=len(rows), unique_observations=len({r['canonical_observation_id'] for r in rows}),
                groups=len(groups), group_balanced_mae=float(np.mean([np.mean(v) for v in groups.values()])),
                pooled_mae=float(abs(e).mean()), source_macro_mae=float(np.mean([np.mean(v) for v in sources.values()])),
                rmse=float(np.sqrt(np.mean(e**2))), median_ae=float(np.median(abs(e))),
                r2=float(1-np.sum(e**2)/den) if den else '', bias=float(e.mean()),
                p90_ae=float(np.quantile(abs(e), .9)), p99_ae=float(np.quantile(abs(e), .99)),
                max_ae=float(abs(e).max()), negative_predictions=int(np.sum(p<0)))


def main():
    if OUT.exists():
        raise RuntimeError('immutable_output_already_exists')
    panel = {r['canonical_observation_id']: r for r in read(PANEL/'panel.csv')}
    outer = read(PANEL/'outer_splits.csv')
    inner = read(PANEL/'inner_splits.csv')
    assert len(panel) == 4515 and len({r['group_id'] for r in panel.values()}) == 165
    expected = set(itertools.product((.1,.25,.5), (20261019,20261020), range(5), TREATMENTS, SEEDS))
    paths = list((STUDY/'qpu_factorial').rglob('terminal_receipt.json'))
    cells = {}
    for p in paths:
        t = json.loads(p.read_text())
        k = (t['fraction'], t['subset_seed'], t['fold'], t['treatment'], t['seed'])
        assert k not in cells
        cells[k] = (p.parent, t)
    assert set(cells) == expected
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in [PANEL/'panel.csv', PANEL/'outer_splits.csv', PANEL/'inner_splits.csv', STUDY/'AGGREGATION_SPEC_V2.md', PARENT/'per_seed_predictions.csv', PARENT/'bootstrap_intervals.csv', PARENT/'aggregation/bootstrap_group_draws.npz']}
    reuse = json.loads((STUDY/'reuse_100_receipt.json').read_text())
    assert sha(PARENT/'per_seed_predictions.csv') == reuse['sha256']
    saved_runner_hash = json.loads((STUDY/'run_manifest.json').read_text())['parent_runner_sha256']
    assert sha(PARENT/'run_qpu_factorial.py') == saved_runner_hash
    all_rows, audits, subset_rows = [], [], []
    paired_identity = {}
    subset_sets = {}
    for k, (d,t) in sorted(cells.items()):
        frac, sub, fold, treatment, seed = k
        ident = json.loads((d/'identity_manifest.json').read_text())
        sr = json.loads((d/'subset_receipt.json').read_text())['subset']
        hist = json.loads((d/'training_history.json').read_text())
        tf = json.loads((d/'transform.json').read_text())
        assert t['status']=='COMPLETE' and len(hist)==500
        assert [x['epoch'] for x in hist]==list(range(1,501))
        assert all(math.isfinite(x['validation_mse']) for x in hist)
        best = min(hist, key=lambda x:(x['validation_mse'],x['epoch']))
        assert (best['epoch'],best['validation_mse'])==(t['best_epoch'],t['best_validation_mse'])
        assert sha(d/'best.pt')==t['checkpoint_sha256']
        test = sorted(r['canonical_observation_id'] for r in outer if int(r['outer_fold'])==fold)
        fit0 = sorted(r['canonical_observation_id'] for r in inner if int(r['outer_fold'])==fold and int(r['inner_fold'])!=0)
        valid = sorted(r['canonical_observation_id'] for r in inner if int(r['outer_fold'])==fold and int(r['inner_fold'])==0)
        groups = sorted({panel[i]['group_id'] for i in fit0})
        random.Random(sub*1009+fold).shuffle(groups)
        selected = set(groups[:max(1,math.ceil(frac*len(groups)))])
        fit = sorted(i for i in fit0 if panel[i]['group_id'] in selected)
        sets = [{panel[i]['group_id'] for i in arr} for arr in (fit, valid, test)]
        assert not any(sets[a]&sets[b] for a,b in ((0,1),(0,2),(1,2)))
        for name, arr in [('fit',fit),('validation',valid),('test',test)]:
            assert sr[name+'_id_sha256']==ids_sha(arr)==ident['partitions'][name]['sha256']
            assert len(arr)==ident['partitions'][name]['count']
        assert sr['selected_group_sha256']==ids_sha(selected)
        assert tf['fit_id_sha256']==ids_sha(fit) and tf['raw_width']==51
        assert tf['global_mean'][37]==0 and tf['global_std'][37]==1
        # Independently recompute fit-only global preprocessing in its declared float32 arithmetic.
        a=np.array([[float(panel[i][f'g{j:02d}']) for j in range(51)] for i in fit],dtype=np.float32)
        assert np.array_equal(a.mean(0), np.asarray(tf['global_mean'],dtype=np.float32))
        sd=a.std(0);sd=np.where(sd==0,1.,sd)
        assert np.array_equal(sd,np.asarray(tf['global_std'],dtype=np.float32))
        rr=read(d/'predictions.csv'); assert len(rr)==len(test)==t['prediction_count']
        assert len({r['canonical_observation_id'] for r in rr})==len(rr)
        assert {r['canonical_observation_id'] for r in rr}==set(test)
        for r in rr:
            source=panel[r['canonical_observation_id']]
            assert float(r['target_seconds'])==float(source['target_seconds'])
            assert r['group_id']==source['group_id'] and r['source_id']==source['source_id']
            assert int(r['outer_fold'])==fold and int(r['seed'])==seed and r['treatment']==treatment
            assert r['status']=='PREDICTED' and math.isfinite(float(r['prediction_seconds']))
            assert r['transform_sha256']==sha(d/'transform.json') and r['checkpoint_sha256']==t['checkpoint_sha256']
            r.update(fraction=f'{frac:.2f}',subset_seed=str(sub))
        all_rows.extend(rr)
        signature=(ident['partitions'],ident['constructor'],ident['batch_policy'],sha(d/'transform.json'),t['parameter_count'])
        pk=(frac,sub,fold,seed,treatment[0])
        if pk in paired_identity: assert paired_identity[pk]==signature
        paired_identity[pk]=signature
        if treatment.endswith('P'):
            arch='MLP' if treatment.startswith('M') else 'GRAPH_MLP'
            source_root=PARENT/f'pretraining/valid_pinned_simple_model/{arch}/seed_{seed}'
            qualified=json.loads((source_root/'terminal_receipt.json').read_text())['checkpoint_sha256']
            # Match the frozen runner's receipt-qualified resolver; an earlier
            # attempted checkpoint may remain at the top-level best.pt path.
            matches=[p for p in source_root.rglob('best.pt') if sha(p)==qualified]
            assert matches and qualified==t['source_checkpoint_sha256']==ident['source_checkpoint_sha256']
        else: assert t['source_checkpoint_sha256'] is None
        for name in ('terminal_receipt.json','identity_manifest.json','training_history.json','transform.json','predictions.csv','best.pt'):
            input_hashes[str((d/name).relative_to(ROOT))]=sha(d/name)
        audits.append(dict(fraction=frac,subset_seed=sub,fold=fold,treatment=treatment,model_seed=seed,epochs=len(hist),best_epoch=t['best_epoch'],best_validation_mse=t['best_validation_mse'],parameter_count=t['parameter_count'],fit_rows=len(fit),validation_rows=len(valid),test_rows=len(test),status='PASS'))
        sk=(frac,sub,fold)
        if sk not in subset_sets:
            subset_sets[sk]=selected
            counts=Counter(panel[i]['source_id'] for i in fit)
            subset_rows.append(dict(fraction=frac,subset_seed=sub,fold=fold,fit_groups=len(selected),full_fit_groups=len(groups),fit_rows=len(fit),validation_rows=len(valid),test_rows=len(test),mali_fit_rows=counts['mali_real_qpu'],qonductor_fit_rows=counts['qonductor_single_circuit_ibm'],qpack_fit_rows=counts['qpack_mcp'],fit_ids_sha256=ids_sha(fit),validation_ids_sha256=ids_sha(valid),test_ids_sha256=ids_sha(test)))
    for sub,fold in itertools.product((20261019,20261020),range(5)):
        assert subset_sets[(.1,sub,fold)]<=subset_sets[(.25,sub,fold)]<=subset_sets[(.5,sub,fold)]
    frozen=read(PARENT/'per_seed_predictions.csv')
    assert len(frozen)==54180
    for r in frozen:r.update(fraction='1.00',subset_seed='FROZEN_A1')
    all_rows.extend(frozen)
    buckets=defaultdict(list)
    for r in all_rows:buckets[(r['fraction'],r['subset_seed'],r['treatment'],r['canonical_observation_id'])].append(r)
    reduced=[]
    for k,rr in sorted(buckets.items()):
        assert len(rr)==3 and {int(r['seed']) for r in rr}==set(SEEDS)
        r={n:rr[0][n] for n in ('fraction','subset_seed','treatment','canonical_observation_id','source_id','group_id','outer_fold','target_seconds')}
        r['prediction_seconds']=float(np.median([float(x['prediction_seconds']) for x in rr]));reduced.append(r)
    assert len(reduced)==126420 and len(all_rows)==379260
    by=defaultdict(list)
    for r in reduced:by[(r['fraction'],r['subset_seed'],r['treatment'])].append(r)
    assert len(by)==28 and all(len(v)==4515 for v in by.values())
    OUT.mkdir()
    write('cell_validation.csv',audits);write('subset_manifest.csv',subset_rows)
    write('all_seed_predictions.csv',all_rows);write('median_predictions.csv',reduced)
    result=[];folds=[];sources=[];seedmetrics=[];tails=[];weight=[]
    for (frac,sub,tr),rr in sorted(by.items()):
        result.append(dict(fraction=frac,subset_seed=sub,treatment=tr,**metrics(rr)))
        for f in range(5):folds.append(dict(fraction=frac,subset_seed=sub,treatment=tr,fold=f,**metrics([r for r in rr if int(r['outer_fold'])==f])))
        for src in sorted({r['source_id'] for r in rr}):sources.append(dict(fraction=frac,subset_seed=sub,treatment=tr,source=src,**metrics([r for r in rr if r['source_id']==src])))
        for r in sorted(rr,key=lambda r:abs(float(r['prediction_seconds'])-float(r['target_seconds'])),reverse=True)[:10]:
            tails.append(dict(**r,absolute_error=abs(float(r['prediction_seconds'])-float(r['target_seconds']))))
    for frac,tr in itertools.product(('0.10','0.25','0.50','1.00'),TREATMENTS):
        rr=[r for r in reduced if r['fraction']==frac and r['treatment']==tr]
        weight.append(dict(fraction=frac,treatment=tr,subset_draws=1 if frac=='1.00' else 2,**metrics(rr)))
    sb=defaultdict(list)
    for r in all_rows:sb[(r['fraction'],r['subset_seed'],r['treatment'],r['seed'])].append(r)
    for (f,s,t,z),rr in sorted(sb.items()):seedmetrics.append(dict(fraction=f,subset_seed=s,treatment=t,model_seed=z,**metrics(rr)))
    for name,rows in [('model_results.csv',weight),('subset_metrics.csv',result),('fold_metrics.csv',folds),('source_metrics.csv',sources),('seed_metrics.csv',seedmetrics),('tail_diagnostics.csv',tails)]:write(name,rows)
    boot=np.load(PARENT/'aggregation/bootstrap_group_draws.npz',allow_pickle=False)
    group_ids=boot['group_ids'];draws=boot['draw_indices'];assert draws.shape==(10000,165)
    assert list(group_ids)==sorted({r['group_id'] for r in panel.values()})
    parent_ci=read(PARENT/'bootstrap_intervals.csv');intervals=[];grouprows=[];strata=[];dist={}
    for frac in ('0.10','0.25','0.50','1.00'):
        subs=['FROZEN_A1'] if frac=='1.00' else ['20261019','20261020']
        for arch,left,right in [('MLP','M-P','M-S'),('GRAPH_MLP','G-P','G-S')]:
            vectors=[]
            for sub in subs:
                aa={r['canonical_observation_id']:r for r in by[(frac,sub,left)]};bb={r['canonical_observation_id']:r for r in by[(frac,sub,right)]};assert aa.keys()==bb.keys()
                gg=defaultdict(list)
                for oid,a in aa.items():
                    b=bb[oid];delta=abs(float(a['prediction_seconds'])-float(a['target_seconds']))-abs(float(b['prediction_seconds'])-float(b['target_seconds']))
                    gg[a['group_id']].append(delta)
                v=np.array([np.mean(gg[g]) for g in group_ids]);vectors.append(v)
                for g,x in zip(group_ids,v):grouprows.append(dict(fraction=frac,subset_seed=sub,architecture=arch,group_id=g,delta_group_mae=float(x)))
                for field in ('outer_fold','source_id'):
                    for val in sorted({a[field] for a in aa.values()}):
                        a=[r for r in aa.values() if r[field]==val];b=[bb[r['canonical_observation_id']] for r in a]
                        strata.append(dict(fraction=frac,subset_seed=sub,architecture=arch,stratum=field,value=val,delta_group_mae=metrics(a)['group_balanced_mae']-metrics(b)['group_balanced_mae']))
            v=np.mean(vectors,axis=0);point=float(v.mean());values=v[draws].mean(1)
            assert abs(point-(next(x for x in weight if x['fraction']==frac and x['treatment']==left)['group_balanced_mae']-next(x for x in weight if x['fraction']==frac and x['treatment']==right)['group_balanced_mae']))<1e-10
            lo,hi=np.quantile(values,[.025,.975]);tail=min(1.,2*min((np.sum(values<=0)+1)/10001,(np.sum(values>=0)+1)/10001))
            if frac=='1.00':
                old=next(x for x in parent_ci if x['contrast']==left+'_minus_'+right)
                assert abs(float(old['point_group_balanced_mae_delta_seconds'])-point)<1e-10
                assert abs(float(old['percentile_95_low'])-lo)<1e-10 and abs(float(old['percentile_95_high'])-hi)<1e-10
            simultaneous=np.quantile(values,[.05/(2*6),1-.05/(2*6)])
            intervals.append(dict(fraction=frac,architecture=arch,contrast=left+' - '+right,delta_group_mae=point,ci95_low=float(lo),ci95_high=float(hi),bootstrap_improvement_fraction=float(np.mean(values<0)),bootstrap_sign_tail_diagnostic=float(tail),holm_adjusted_diagnostic='',sensitivity_bonferroni_low=float(simultaneous[0]) if frac!='1.00' else '',sensitivity_bonferroni_high=float(simultaneous[1]) if frac!='1.00' else '',groups=165,subset_draws=len(subs),replicates=10000,bootstrap_seed=42,inference_scope='conditional_saved_runs' if frac!='1.00' else 'replayed_original_A1'))
            dist[f'{frac}_{arch}']=values
    new=sorted([x for x in intervals if x['fraction']!='1.00'],key=lambda x:x['bootstrap_sign_tail_diagnostic']);last=0
    for i,x in enumerate(new):last=max(last,min(1.,(6-i)*x['bootstrap_sign_tail_diagnostic']));x['holm_adjusted_diagnostic']=last
    write('paired_comparisons.csv',intervals);write('paired_group_deltas.csv',grouprows);write('stratified_paired_deltas.csv',strata)
    np.savez_compressed(OUT/'bootstrap_distributions.npz',group_ids=group_ids,draw_indices=draws,**dist)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,arch in zip(axs,('MLP','GRAPH_MLP')):
        rr=[r for r in intervals if r['architecture']==arch];x=[100*float(r['fraction']) for r in rr];y=[r['delta_group_mae'] for r in rr]
        ax.errorbar(x,y,yerr=[[r['delta_group_mae']-r['ci95_low'] for r in rr],[r['ci95_high']-r['delta_group_mae'] for r in rr]],fmt='o-');ax.axhline(0,color='gray',ls='--');ax.set(title=arch,xlabel='Fit groups retained (%)',ylabel='Pretrained - scratch group MAE (s)');ax.set_xticks([10,25,50,100])
    fig.tight_layout();fig.savefig(OUT/'transfer_effect.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for tr in TREATMENTS:
        rr=[r for r in weight if r['treatment']==tr];ax.plot([float(r['fraction'])*100 for r in rr],[r['group_balanced_mae'] for r in rr],'o-',label=tr)
    ax.set(xlabel='Fit groups retained (%)',ylabel='Group-balanced MAE (s)');ax.legend();fig.tight_layout();fig.savefig(OUT/'absolute_performance.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for row,arch in enumerate(('MLP','GRAPH_MLP')):
        for col,field in enumerate(('outer_fold','source_id')):
            ax=axs[row,col]
            for val in sorted({r['value'] for r in strata if r['stratum']==field}):
                ys=[]
                for f in ('0.10','0.25','0.50','1.00'):
                    ys.append(float(np.mean([r['delta_group_mae'] for r in strata if r['fraction']==f and r['architecture']==arch and r['stratum']==field and r['value']==val])))
                ax.plot([10,25,50,100],ys,'o-',label=val)
            ax.axhline(0,color='gray',ls='--');ax.set(title=f'{arch}: {field}',ylabel='Delta group MAE (s)',xlabel='Fit groups (%)');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'heterogeneity.png',dpi=180);plt.close(fig)
    receipt=dict(status='PASS',scope='saved_prediction_aggregation_only',training_cells=360,seed_prediction_rows=len(all_rows),median_prediction_rows=len(reduced),unique_test_observations=4515,structural_groups=165,all_epoch_checks=True,all_paired_transform_checks=True,fit_only_global_preprocessing_replayed=True,all_group_leakage_checks=True,parent_100_predictions_and_intervals_replayed=True,checkpoint_byte_hashes_verified=True,checkpoint_tensor_inference_replayed=False,created_utc=datetime.now(timezone.utc).isoformat(),implementation_sha256=sha(Path(__file__)),input_hashes=input_hashes)
    (OUT/'validation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='input_hashes'},indent=2))
    print(json.dumps(intervals,indent=2))


if __name__=='__main__':main()
