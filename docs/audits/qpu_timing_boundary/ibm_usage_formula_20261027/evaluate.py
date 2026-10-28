"""Fixed IBM formulas on saved Track-A inputs; standard library only, no fitting."""
import csv
import hashlib
import json
import math
import platform
import statistics as st
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
PANEL = ROOT / 'artifacts/real_qpu/qualified_logical_panel/panel.csv'
TIMING = ROOT / 'artifacts/postfreeze_track_a_transpiled_representation_v1/features/timing_features.csv'
PROVENANCE = ROOT / 'artifacts/postfreeze_track_a_transpiled_representation_v1/p0_coverage/observation_coverage.csv'

def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

def percentile(values, q):
    v = sorted(values)
    x = (len(v)-1)*q
    i = math.floor(x)
    j = math.ceil(x)
    return v[i] + (v[j]-v[i])*(x-i)

def metrics(rows, method, scope):
    y = [r['target_seconds'] for r in rows]
    errors = [r[method]-r['target_seconds'] for r in rows]
    ae = [abs(e) for e in errors]
    groups, sources = defaultdict(list), defaultdict(list)
    for r, e in zip(rows, ae):
        groups[r['group_id']].append(e)
        sources[r['source_id']].append(e)
    sst = sum((v-st.mean(y))**2 for v in y)
    return dict(method=method, scope=scope, rows=len(rows), groups=len(groups),
        mae_seconds=st.mean(ae), group_balanced_mae_seconds=st.mean(st.mean(v) for v in groups.values()),
        source_macro_mae_seconds=st.mean(st.mean(v) for v in sources.values()),
        rmse_seconds=math.sqrt(st.mean(e*e for e in errors)),
        raw_r2=1-sum(e*e for e in errors)/sst if sst else '',
        bias_seconds=st.mean(errors), median_absolute_error_seconds=st.median(ae),
        p90_absolute_error_seconds=percentile(ae,.9), p99_absolute_error_seconds=percentile(ae,.99),
        max_absolute_error_seconds=max(ae))

def main():
    panel, timing, provenance = read(PANEL), read(TIMING), read(PROVENANCE)
    ids = [r['canonical_observation_id'] for r in panel]
    assert len(panel) == len(set(ids)) == 4515
    assert len({r['group_id'] for r in panel}) == 165
    assert Counter(r['source_id'] for r in panel) == {'mali_real_qpu':340, 'qonductor_single_circuit_ibm':230, 'qpack_mcp':3945}
    def index(rows):
        d = {r['canonical_observation_id']:r for r in rows}
        assert len(d) == len(rows) and set(d) == set(ids)
        return d
    t, p = index(timing), index(provenance)
    rows = []
    for r in panel:
        rid = r['canonical_observation_id']
        y, shots, duration = float(r['target_seconds']), float(r['shots']), float(t[rid]['nominal_single_shot_duration_seconds'])
        assert all(math.isfinite(v) and v > 0 for v in (y, shots, duration))
        assert shots == int(shots)
        assert abs(float(t[rid]['log1p_shots'])-math.log1p(shots)) < 1e-12
        assert p[rid]['group_id'] == r['group_id'] and p[rid]['source_id'] == r['source_id']
        rows.append(dict(canonical_observation_id=rid, source_id=r['source_id'], group_id=r['group_id'],
            backend=r['backend'], shots=int(shots), target_seconds=y,
            nominal_single_shot_duration_seconds=duration,
            structure_provenance=p[rid]['structure_provenance'], timing_provenance=p[rid]['timing_provenance'],
            IBM_QUICK=2+.00035*shots, IBM_DURATION_DEFAULT_DELAY=2+shots*(duration+.00025)))
    result=[]
    for method in ('IBM_QUICK','IBM_DURATION_DEFAULT_DELAY'):
        result.append(metrics(rows, method, 'all'))
        for source in sorted({r['source_id'] for r in rows}):
            result.append(metrics([r for r in rows if r['source_id']==source], method, source))
    write('predictions.csv', rows)
    write('metrics.csv', result)
    receipt=dict(status='PASS', rows=len(rows), groups=165, exact_id_join='PASS',
        source_counts=dict(Counter(r['source_id'] for r in rows)),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        python=platform.python_version(), no_fit=True, no_circuit_execution=True,
        input_sha256={str(x.relative_to(ROOT)):digest(x) for x in (PANEL,TIMING,PROVENANCE,OUT/'PROTOCOL.md',Path(__file__))},
        output_sha256={name:digest(OUT/name) for name in ('predictions.csv','metrics.csv')})
    (OUT/'validation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
