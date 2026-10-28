#!/usr/bin/env python3
"""Public aggregate/hash verification; no model, circuit, calibration or training."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def rows(rel):
 with (ROOT/rel).open(newline='') as f: return list(csv.DictReader(f))

def main():
 index=json.loads((ROOT/'docs/publication/RECENT_QPU_INDEX.json').read_text())
 assert index['scope']=='aggregate_reports_and_source_only'
 assert not index['private_qpu_rows_included']
 paths=[r['path'] for r in index['files']]; assert len(paths)==len(set(paths))
 for entry in index['files']:
  p=ROOT/entry['path']; assert p.is_file() and not p.is_symlink()
  assert p.stat().st_size==entry['bytes'],entry['path']
  assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['published_sha256'],entry['path']
  if p.suffix=='.py': compile(p.read_text(),str(p),'exec')
 base='artifacts/postfreeze_mali_shots_ablation_v1/'
 scores={r['treatment']:r for r in rows(base+'aggregate/metrics.csv')}
 assert set(scores)=={'M0','M1','G0','G1'}
 for r in scores.values(): assert int(r['rows'])==4515 and int(r['unique_groups'])==165
 pairs=rows(base+'aggregate/paired_comparisons.csv'); assert len(pairs)==2
 for r in pairs:
  delta=float(scores[r['treatment']]['group_balanced_mae_seconds'])-float(scores[r['control']]['group_balanced_mae_seconds'])
  assert math.isclose(delta,float(r['delta_group_balanced_mae_seconds']),abs_tol=1e-12)
  assert float(r['ci95_low'])<0<float(r['ci95_high'])
  assert int(r['bootstrap_replicates'])==10000 and r['bootstrap_unit']=='structural_group'
 receipt=json.loads((ROOT/(base+'completion_receipt.json')).read_text())
 assert receipt['status']=='COMPLETE_VALIDATED' and receipt['completed_cells']==30
 audit=json.loads((ROOT/(base+'audit_preprocessing_v1/receipt.json')).read_text())
 assert audit['status']=='PASS' and audit['checkpoint_replays']==30
 print(f'PASS: {len(paths)} file hashes/sizes, Python syntax, shots aggregate deltas/coverage/CIs and saved completion metadata. No private prediction replay or model execution.')

if __name__=='__main__': main()
