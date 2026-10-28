"""Saved-artifact diagnostics only. No simulator calls, fitting or new splits."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PRED = ROOT/'artifacts/real_qpu/qualified_logical_panel/summary/oof_predictions.csv'
FEAT = ROOT/'artifacts/postfreeze_notads_hyperq_track_a_v1/features.csv'
NPRED = FEAT.parent/'predictions.csv'
REG = ROOT/'artifacts/benchmark_v3/recovery_real_qpu_20260929_v3/c133_snapshot_registry_v3.csv'
METHODS = ['mali_logical_graph','mali_global_mlp','qonductor_budgeted_selector',
 'qcre_source_estimator__raw','qcre_source_estimator_shot_scaled__raw',
 'qcre_source_estimator_shot_scaled__affine','qiskit_estimate_duration_snapshot__raw',
 'qiskit_shot_scaled_schedule_seconds__raw','qiskit_shot_scaled_schedule_seconds__affine',
 'hyb_nominal_r2_effective_cost_v1__raw','hyb_nominal_r2_log_cost_ridge_v1']

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def corr(x,y):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0: return (None,None)
    return float(pearsonr(x/np.max(abs(x)),y/np.max(abs(y))).statistic),float(spearmanr(x,y).statistic)

def main():
    a=pd.read_csv(PRED,low_memory=False)
    ids=a[['canonical_observation_id','actual_seconds','source_id','group_id','outer_fold','backend']].drop_duplicates()
    assert len(ids)==4515 and ids.canonical_observation_id.is_unique and ids.group_id.nunique()==165
    f=pd.read_csv(FEAT)
    f=f.merge(ids,on=['canonical_observation_id','source_id','group_id','outer_fold','backend'],validate='one_to_one')
    reg=pd.read_csv(REG).set_index('snapshot_id')
    pins={}
    inv=[]
    for (backend,sid),g in f.groupby(['backend','snapshot_id']):
        r=reg.loc[sid];path=Path(r.configuration_path)
        assert digest(path)==r.configuration_sha256
        j=json.loads(path.read_text());t=j['processor_type']
        generation=t['family']+' r'+str(t['revision'])+str(t.get('segment',''))
        pins[str(path)]=digest(path)
        inv.append(dict(backend=backend,snapshot_id=sid,family=t['family'],generation=generation,
            hardware_qubits=j['n_qubits'],basis_gates=';'.join(j['basis_gates']),
            rows=len(g),groups=g.group_id.nunique(),sources=';'.join(sorted(g.source_id.unique())),
            snapshot_date=r.snapshot_or_update_date,config_path=str(path),config_sha256=digest(path)))
    inventory=pd.DataFrame(inv);inventory.to_csv(OUT/'backend_inventory.csv',index=False)
    f=f.merge(inventory[['backend','snapshot_id','family','generation']],on=['backend','snapshot_id'],validate='many_to_one')
    f.groupby(['family','generation','backend','source_id']).agg(rows=('group_id','size'),groups=('group_id','nunique')).reset_index().to_csv(OUT/'source_generation_coverage.csv',index=False)
    b=pd.read_csv(NPRED).rename(columns={'method':'method_id','target_seconds':'actual_seconds','prediction_seconds':'predicted_seconds'})
    b=b[b.method_id.isin(['notads_code_shots_affine','notads_paper_shots_affine'])]
    b=b.merge(ids[['canonical_observation_id','backend']],on='canonical_observation_id',validate='many_to_one')
    cols=['canonical_observation_id','method_id','actual_seconds','predicted_seconds','status']
    p=pd.concat([a[a.method_id.isin(METHODS)][cols],b[cols]],ignore_index=True)
    assert not p.duplicated(['canonical_observation_id','method_id']).any()
    p=p.merge(f.drop(columns='actual_seconds'),on='canonical_observation_id',validate='many_to_one')
    assert (p.actual_seconds==p.canonical_observation_id.map(ids.set_index('canonical_observation_id').actual_seconds)).all()
    result=[]
    strata=[[],['family'],['generation'],['backend'],['source_id'],['source_id','backend'],['source_id','generation']]
    for method,pm in p.groupby('method_id'):
        for keys in strata:
            groups=[('ALL',pm)] if not keys else pm.groupby(keys)
            for key,assigned in groups:
                ok=assigned[(assigned.status=='predicted')&np.isfinite(assigned.predicted_seconds)]
                row={'method':method,'stratum':'+'.join(keys) or 'ALL','stratum_value':str(key),
                    'assigned_rows':len(assigned),'valid_rows':len(ok),'groups':ok.group_id.nunique()}
                if len(ok):
                    x=ok.predicted_seconds.to_numpy();y=ok.actual_seconds.to_numpy();e=x-y
                    r,rho=corr(x,y)
                    row.update(mae=float(np.mean(abs(e))),bias=float(np.mean(e)),
                        group_balanced_mae=float(ok.assign(error=abs(e)).groupby('group_id').error.mean().mean()),
                        target_median=float(np.median(y)),prediction_median=float(np.median(x)),
                        median_prediction_target_ratio=float(np.median(x/y)),
                        underprediction_fraction=float(np.mean(e<0)),p90_error=float(np.quantile(abs(e),.9)),
                        pearson=r,spearman=rho)
                result.append(row)
    metrics=pd.DataFrame(result);metrics.to_csv(OUT/'method_stratum_metrics.csv',index=False)
    # Validate overall MAEs against the original saved metric artifact.
    saved=pd.read_csv(PRED.parent/'method_metrics.csv').set_index('method_id')
    for row in result:
        if row['stratum']=='ALL' and row['method'] in METHODS:
            assert row['valid_rows']==saved.loc[row['method'],'predicted_rows']
            assert np.isclose(row['mae'],saved.loc[row['method'],'mae_seconds'],atol=1e-12)
    q=f[f.source_id=='qpack_mcp'].copy()
    q['predicted_shots_seconds']=q.hyperq_full_seconds*q.shots
    q['prediction_target_ratio']=q.predicted_shots_seconds/q.actual_seconds
    q.to_csv(OUT/'qpack_row_diagnostics.csv',index=False)
    cells=[]
    for (backend,physical),g in q.groupby(['backend','physical_hash']):
        assert g.predicted_shots_seconds.nunique()==1 and g.shots.nunique()==1
        cells.append(dict(backend=backend,physical_hash=physical,rows=len(g),groups=g.group_id.nunique(),
            folds=';'.join(str(v) for v in sorted(g.outer_fold.unique())),
            target_min=g.actual_seconds.min(),target_median=g.actual_seconds.median(),target_max=g.actual_seconds.max(),
            target_std=g.actual_seconds.std(ddof=0),prediction=g.predicted_shots_seconds.iloc[0],
            within_sse=float(np.sum((g.actual_seconds-g.actual_seconds.mean())**2))))
    c=pd.DataFrame(cells);c.to_csv(OUT/'qpack_constant_input_cells.csv',index=False)
    q.groupby(['group_id','backend']).agg(rows=('actual_seconds','size'),
        target_mean=('actual_seconds','mean'),target_median=('actual_seconds','median'),
        prediction=('predicted_shots_seconds','first')).reset_index().to_csv(OUT/'qpack_group_backend.csv',index=False)
    total=float(np.sum((q.actual_seconds-q.actual_seconds.mean())**2))
    backend_means=q.groupby('backend').actual_seconds.transform('mean')
    summary=dict(rows=len(q),groups=q.group_id.nunique(),backends=q.backend.nunique(),
        graph_backend_cells=len(c),constant_cells_with_multiple_rows=int((c.rows>1).sum()),
        target_min=float(q.actual_seconds.min()),target_median=float(q.actual_seconds.median()),target_max=float(q.actual_seconds.max()),
        prediction_min=float(q.predicted_shots_seconds.min()),prediction_median=float(q.predicted_shots_seconds.median()),prediction_max=float(q.predicted_shots_seconds.max()),
        ratio_median=float(q.prediction_target_ratio.median()),within_cell_fraction_total_target_sst=float(c.within_sse.sum()/total),
        between_backend_fraction_total_target_sst=float(np.sum((backend_means-q.actual_seconds.mean())**2)/total),
        widest_cell=c.loc[(c.target_max-c.target_min).idxmax()].to_dict())
    (OUT/'qpack_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'validation.json').write_text(json.dumps(dict(status='PASS',rows=4515,groups=165,
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_hashes={str(x.relative_to(ROOT)):digest(x) for x in [PRED,FEAT,NPRED,REG,PRED.parent/'method_metrics.csv']},
        config_hashes=pins,script_sha256=digest(Path(__file__)),
        original_mae_replay='PASS',no_fitting=True,no_execution=True),indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
