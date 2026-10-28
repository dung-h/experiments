"""Independent saved-output replay, using centered scalar regression and fsum."""
import csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/postfreeze_notads_hyperq_track_a_v1'
def read(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ids_digest(ids):return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()

def main():
    panel=read(ROOT/'artifacts/real_qpu/qualified_logical_panel/panel.csv')
    meta={r['canonical_observation_id']:r for r in panel}
    splits={r['canonical_observation_id']:r for r in read(ROOT/'artifacts/real_qpu/qualified_logical_panel/outer_splits.csv')}
    features={r['canonical_observation_id']:r for r in read(OUT/'features.csv')}
    rows=read(OUT/'predictions.csv');by=defaultdict(dict);checks=0
    for r in rows:
        rid=r['canonical_observation_id'];m=meta[rid]
        assert rid not in by[r['method']]
        assert float(r['target_seconds'])==float(m['target_seconds'])
        assert r['group_id']==m['group_id'] and r['outer_fold']==splits[rid]['outer_fold']
        assert math.isfinite(float(r['prediction_seconds'])) and float(r['prediction_seconds'])>=0
        by[r['method']][rid]=r;checks+=4
    assert len(by)==9 and len(rows)==4515*9
    for v in by.values():assert set(v)==set(meta)
    for fit in read(OUT/'calibration_fits.csv'):
        method=fit['method'];fold=fit['fold'];route_scale=method.removesuffix('_affine');route,scale=route_scale.rsplit('_',1)
        field='hyperq_full_seconds' if route=='hyperq_paper' else route
        train=sorted(i for i in meta if splits[i]['outer_fold']!=fold);test=sorted(set(meta)-set(train))
        assert not {meta[i]['group_id'] for i in train}&{meta[i]['group_id'] for i in test}
        assert ids_digest(train)==fit['train_id_sha256'] and ids_digest(test)==fit['test_id_sha256']
        def x(i):return float(features[i][field])*(int(float(meta[i]['shots'])) if scale=='shots' else 1)
        xx=np.array([x(i) for i in train]);yy=np.array([float(meta[i]['target_seconds']) for i in train])
        dx=xx-xx.mean();dy=yy-yy.mean()
        b=float(np.dot(dx,dy)/np.dot(dx,dx));a=float(yy.mean()-b*xx.mean())
        for i in test:
            expected=max(0,a+b*x(i));actual=float(by[method][i]['prediction_seconds'])
            assert math.isclose(expected,actual,rel_tol=1e-8,abs_tol=1e-7),(method,fold,i,expected,actual)
            checks+=1
    for s in read(OUT/'metrics.csv'):
        rr=list(by[s['method']].values())
        if s['scope'].startswith('fold_'):rr=[r for r in rr if r['outer_fold']==s['scope'][5:]]
        elif s['scope']!='all':rr=[r for r in rr if r['source_id']==s['scope']]
        errors=[float(r['prediction_seconds'])-float(r['target_seconds']) for r in rr];n=len(rr)
        groups=defaultdict(list)
        for r,e in zip(rr,errors):groups[r['group_id']].append(abs(e))
        yy=[float(r['target_seconds']) for r in rr];mean=math.fsum(yy)/n
        values={'mae_seconds':math.fsum(abs(e) for e in errors)/n,
            'group_balanced_mae_seconds':math.fsum(math.fsum(v)/len(v) for v in groups.values())/len(groups),
            'rmse_seconds':math.sqrt(math.fsum(e*e for e in errors)/n),
            'raw_r2':1-math.fsum(e*e for e in errors)/math.fsum((y-mean)**2 for y in yy)}
        for k,v in values.items():assert math.isclose(v,float(s[k]),rel_tol=1e-10,abs_tol=1e-10),(s['method'],s['scope'],k)
        checks+=4
    z=np.load(OUT/'bootstrap_draws.npz');groups=z['groups'].tolist();draws=z['draws']
    assert np.array_equal(draws,np.random.default_rng(42).integers(0,165,size=(10000,165)))
    for pair in read(OUT/'paired_comparisons.csv'):
        differences=[]
        for g in groups:
            ids=[i for i in meta if meta[i]['group_id']==g]
            def err(method):return math.fsum(abs(float(by[method][i]['prediction_seconds'])-float(meta[i]['target_seconds'])) for i in ids)/len(ids)
            differences.append(err(pair['method_a'])-err(pair['method_b']))
        delta=np.array(differences);boot=delta[draws].mean(axis=1)
        for k,v in [('delta_group_balanced_mae_seconds',delta.mean()),('ci95_low_seconds',np.quantile(boot,.025)),('ci95_high_seconds',np.quantile(boot,.975))]:
            assert math.isclose(float(pair[k]),float(v),rel_tol=1e-10,abs_tol=1e-10)
            checks+=1
    # Compare reimplemented HyperQ against existing frozen Qiskit predictions.
    frozen=read(ROOT/'artifacts/real_qpu/qualified_logical_panel/analytical/attempts.csv')
    mapping={'qiskit_estimate_duration_snapshot__raw':'hyperq_paper_raw',
             'qiskit_shot_scaled_schedule_seconds__raw':'hyperq_paper_shots',
             'qiskit_estimate_duration_snapshot__affine':'hyperq_paper_raw_affine',
             'qiskit_shot_scaled_schedule_seconds__affine':'hyperq_paper_shots_affine'}
    parity=[]
    for old,new in mapping.items():
        rr=[r for r in frozen if r['method_id']==old];assert len(rr)==4515
        diffs=[]
        for r in rr:
            rid=r['canonical_observation_id'];a=float(r['predicted_seconds']);b=float(by[new][rid]['prediction_seconds'])
            assert float(r['actual_seconds'])==float(meta[rid]['target_seconds'])
            assert math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-7),(new,rid,a,b)
            diffs.append(abs(a-b));checks+=2
        parity.append({'new_method':new,'existing_method':old,'rows':4515,'max_absolute_prediction_difference_seconds':max(diffs),'status':'PASS'})
    receipt=json.loads((OUT/'feature_receipt.json').read_text())
    for rel,h in receipt['inputs_sha256'].items():assert digest(ROOT/rel)==h
    out={'status':'PASS','checks':checks,'prediction_rows':len(rows),'methods':9,'observations_per_method':4515,
         'group_count':165,'centered_OLS_independent_replay':'PASS','metric_and_bootstrap_replay':'PASS',
         'original_input_hashes_unchanged':'PASS','HyperQ_existing_Qiskit_prediction_parity':parity,
         'source_sha256':digest(__file__)}
    (OUT/'independent_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
