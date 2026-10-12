#!/usr/bin/env python3
"""Export saved full-grid aggregates only, without private row ledgers."""
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/real_qpu/generalization/complete_grid_v1/results/full_grid_20261011'
DEST=ROOT/'work/completed-science-publication'
FILES={
 'case_status.csv':'case_status.csv','seed_metrics.csv':'seed_metrics.csv',
 'method_metrics.csv':'method_metrics.csv','selector_identity_audit.csv':'selector_identity_audit.csv',
 'reader_tables/median_metrics.csv':'median_metrics.csv','reader_tables/train_references.csv':'train_references.csv',
 'reader_tables/input_support.csv':'input_support.csv','reader_tables/feature_masking.csv':'feature_masking.csv',
 'reader_tables/source_metrics.csv':'source_metrics.csv','reader_tables/distribution_shift.csv':'distribution_shift.csv',
}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 records=[]
 for src,name in FILES.items():
  p=SOURCE/src
  with p.open() as f:header=next(csv.reader(f))
  assert not {'canonical_observation_id','actual_seconds','predicted_seconds','test_id','fit_id','group_id'}.intersection(header)
  dest=DEST/'results/generalization/full_grid'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
  records.append(dict(source_path=str(p.relative_to(ROOT)),source_sha256=sha(p),public_path=str(dest.relative_to(DEST)),published_sha256=sha(dest),transformation='exact aggregate CSV bytes'))
 p=SOURCE/'reader_tables/paired_comparisons.csv';before=sha(p)
 with p.open() as f:
  reader=csv.DictReader(f);headers=reader.fieldnames;kept=[]
  neural={'mali_logical_graph','mali_global_mlp'};references={'train_global_median','train_source_median','qonductor_budgeted_selector'}
  for row in reader:
   pair={row['method_a'],row['method_b']}
   if row['slice_axis']=='support_scope' and row['scope']=='all_test' and pair.intersection(neural) and pair.issubset(neural|references):kept.append(row)
 assert len({r['case_id'] for r in kept if {r['method_a'],r['method_b']}==neural})==21
 dest=DEST/'results/generalization/full_grid/paired_comparisons.csv'
 s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=headers,lineterminator='\n');w.writeheader();w.writerows(kept);dest.write_text(s.getvalue())
 assert sha(p)==before
 records.append(dict(source_path=str(p.relative_to(ROOT)),source_sha256=before,public_path=str(dest.relative_to(DEST)),published_sha256=sha(dest),
  transformation='exact saved-row selection: support_scope/all_test, neural vs neural/train-only references/CPU selector',selected_rows=len(kept)))
 tool=Path(__file__).resolve();target=DEST/'methods/research_support/export_complete_grid_publication.py';target.write_bytes(tool.read_bytes())
 records.append(dict(source_path=str(tool.relative_to(ROOT)),source_sha256=sha(tool),public_path=str(target.relative_to(DEST)),published_sha256=sha(target),transformation='exact controlled-export source'))
 manifest=dict(scope='Full 21-case aggregate supplement; no observation-level predictions or group ledgers',files=records,
   CPU_cases=21,neural_cases=21,seed_metric_rows=126,private_row_level_QPU_included=False,
   excluded=['neural_seed_attempts.csv','seed_dispersion.csv','group_metrics.csv','reader_tables/group_metrics.csv','complete58MBpaired_registry'],
   values_changed=False,new_experiments=False)
 (DEST/'provenance/complete_grid_publication.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'status':'PASS','files':len(records),'selected_saved_pairs':len(kept),'source_values_changed':False},indent=2))
if __name__=='__main__':main()
