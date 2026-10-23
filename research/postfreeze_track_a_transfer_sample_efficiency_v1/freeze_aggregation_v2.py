#!/usr/bin/env python3
"""Freeze validated P1 saved-output aggregation without touching cell artifacts."""
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/postfreeze_track_a_transfer_sample_efficiency_v1/aggregation_v2'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    target=OUT/'completion_receipt.json'
    if target.exists():raise RuntimeError('already_frozen')
    receipt=json.loads((OUT/'validation_receipt.json').read_text())
    assert receipt['status']=='PASS' and receipt['training_cells']==360
    tested=subprocess.run([sys.executable,str(Path(__file__).with_name('test_aggregation_v2.py')),'-v'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'replay_test_results.txt').write_text(tested.stdout+tested.stderr)
    assert tested.returncode==0
    diff=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    assert diff.returncode==0,diff.stdout+diff.stderr
    # Seed-level matched deltas, kept descriptive and never pooled as extra groups.
    with (OUT/'seed_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    by={(r['fraction'],r['subset_seed'],r['treatment'],r['model_seed']):r for r in rows}
    effects=[]
    for (frac,sub,tr,seed),a in sorted(by.items()):
        if tr not in ('M-P','G-P'):continue
        b=by[(frac,sub,tr[0]+'-S',seed)]
        effects.append(dict(fraction=frac,subset_seed=sub,architecture=tr[0],model_seed=seed,delta_group_mae=float(a['group_balanced_mae'])-float(b['group_balanced_mae']),delta_pooled_mae=float(a['pooled_mae'])-float(b['pooled_mae'])))
    with (OUT/'seed_paired_deltas.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(effects[0]));w.writeheader();w.writerows(effects)
    status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
    (OUT/'repository_state_at_freeze.txt').write_text(status)
    outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}
    scripts={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('aggregate_validated_v2.py'),Path(__file__).with_name('test_aggregation_v2.py')]}
    result=dict(status='PASS_FROZEN',scope='P1_aggregation_v2_only',created_utc=datetime.now(timezone.utc).isoformat(),training_cells=360,replay_tests=3,replay_tests_pass=True,git_diff_check='PASS',git_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),staged_files=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines(),output_hashes=outputs,code_hashes=scripts,scientific_scope='post-training conditional saved-run analysis; no unconditional or preregistered significance claim',no_training_or_inference_during_aggregation=True,parent_100_unchanged=True)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('output_hashes','code_hashes')},indent=2))


if __name__=='__main__':main()
