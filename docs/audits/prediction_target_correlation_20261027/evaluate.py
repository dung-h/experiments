"""Read saved predictions only; no fitting, circuit execution or split changes."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/real_qpu/qualified_logical_panel/summary/oof_predictions.csv'
NEW = ROOT / 'artifacts/postfreeze_notads_hyperq_track_a_v1/predictions.csv'
FEATURES = NEW.parent / 'features.csv'

def main():
    a = pd.read_csv(BASE, low_memory=False).rename(columns={
        'method_id':'method', 'actual_seconds':'target', 'predicted_seconds':'value'})
    ids = a[['canonical_observation_id','source_id','group_id','outer_fold','target']].drop_duplicates()
    assert len(ids) == 4515 and ids.canonical_observation_id.is_unique
    assert ids.group_id.nunique() == 165
    b = pd.read_csv(NEW).rename(columns={'target_seconds':'target','prediction_seconds':'value'})
    check = b.merge(ids, on='canonical_observation_id', suffixes=('','_base'), validate='many_to_one')
    for col in ['source_id','group_id','outer_fold','target']:
        assert (check[col] == check[col+'_base']).all(), col
    f = pd.read_csv(FEATURES).merge(ids, on=['canonical_observation_id','source_id','group_id','outer_fold'], validate='one_to_one')
    assert len(f) == 4515
    raw = []
    for name in ['notads_paper','notads_code']:
        for shots in [False, True]:
            t = f[ids.columns].copy()
            t['method'] = name + ('_shots_score' if shots else '_raw_score')
            t['value'] = f[name] * (f.shots if shots else 1)
            t['status'] = 'predicted'
            raw.append(t)
    cols = ['canonical_observation_id','source_id','group_id','outer_fold','method','target','value','status']
    data = pd.concat([a[cols], b[cols]] + [x[cols] for x in raw], ignore_index=True)
    assert not data.duplicated(['method','canonical_observation_id']).any()
    rows = []
    for method, all_rows in data.groupby('method'):
        for source in ['ALL'] + sorted(ids.source_id.unique()):
            assigned = all_rows if source == 'ALL' else all_rows[all_rows.source_id == source]
            valid = assigned[(assigned.status == 'predicted') & np.isfinite(assigned.value) & np.isfinite(assigned.target)]
            x, y = valid.value.to_numpy(float), valid.target.to_numpy(float)
            ok = len(x) > 2 and np.ptp(x) > 0 and np.ptp(y) > 0
            # Positive rescaling prevents squared-sum overflow; correlation is unchanged.
            r = float(pearsonr(x / np.max(abs(x)), y / np.max(abs(y))).statistic) if ok else None
            rho = float(spearmanr(x,y).statistic) if ok else None
            rows.append(dict(method=method,source_id=source,assigned_rows=len(assigned),
                valid_rows=len(valid),groups=valid.group_id.nunique(),pearson_r=r,spearman_rho=rho,
                weighting='pooled observations',status='OK' if ok else 'UNDEFINED'))
    pd.DataFrame(rows).to_csv(OUT/'correlations.csv',index=False)
    receipt = dict(status='PASS',rows=4515,groups=165,methods=data.method.nunique(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE,NEW,FEATURES]},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_fitting=True,no_circuit_execution=True,no_bootstrap_inference=True)
    (OUT/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__ == '__main__':
    main()
