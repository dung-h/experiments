#!/usr/bin/env python3
"""Explicit aggregate/code export; never walks a scientific artifact directory."""
import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

GROUPS = {
 'docs/publication': ['RECENT_QPU_FINDINGS.md'],
 'docs/literature_review': ['PRESENTATION_PLAN.md'],
 'docs/audits/qpu_timing_boundary': ['RESULT_TIME_TAKEN_PROVENANCE_NOTE.md'],
 'docs/audits/qpu_timing_boundary/ibm_usage_formula_20261027':
  ['PROTOCOL.md','REPORT.md','metrics.csv','validation_receipt.json','independent_verification.json','evaluate.py'],
 'docs/audits/backend_generation_qpack_20261027':
  ['REPORT.md','backend_inventory.csv','source_generation_coverage.csv','method_stratum_metrics.csv','validation.json','evaluate.py'],
 'docs/audits/prediction_target_correlation_20261027':
  ['REPORT.md','correlations.csv','fold_rank_check.csv','validation.json','evaluate.py'],
 'docs/audits/notads_hyperq_estimator_20261027': ['REPORT.md','source_inventory.json'],
 'artifacts/postfreeze_notads_hyperq_track_a_v1':
  ['PROTOCOL.md','PROTOCOL_AMENDMENT_01.md','README.md','REPORT.md','metrics.csv','paired_comparisons.csv','completion_receipt.json','final_validation_receipt.json'],
 'artifacts/postfreeze_mali_shots_ablation_v1': ['PROTOCOL.md','FINAL_REPORT.md','completion_receipt.json'],
 'artifacts/postfreeze_mali_shots_ablation_v1/aggregate':
  ['metrics.csv','paired_comparisons.csv','fold_group_balanced_effects.csv','slice_metrics.csv','seed_metrics.csv'],
 'artifacts/postfreeze_mali_shots_ablation_v1/audit_preprocessing_v1': ['receipt.json'],
 'research/postfreeze_mali_shots_ablation_v1':
  ['run.py','shots_core.py','aggregate.py','preflight.py','lock_protocol.py','validate.py','audit_preprocessing.py','analyze_failures.py'],
 'research/postfreeze_notads_hyperq_track_a_v1': ['run.py','path_kernel.cpp','validate_saved.py'],
 'scripts': ['export_recent_qpu_findings.py','verify_recent_qpu_findings.py'],
}
FORBIDDEN_COLUMNS = {'canonical_observation_id','observation_id','row_id','group_id',
 'source_qasm_sha256','raw_qasm_sha256','target_seconds','actual_seconds',
 'predicted_seconds','prediction_seconds','test_id','fit_id'}

def sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
 parser.add_argument('--destination',type=Path,required=True)
 a=parser.parse_args(); source=a.source.resolve(); dest=a.destination.resolve()
 if source==dest or not (dest/'.git').exists(): raise ValueError('separate_git_checkout_required')
 paths=sorted(f'{folder}/{name}' for folder,names in GROUPS.items() for name in names)
 assert len(paths)==len(set(paths))
 records=[]; payloads={}
 for rel in paths:
  p=source/rel; q=dest/rel
  if not p.is_file() or p.is_symlink(): raise ValueError(f'missing_or_symlink:{rel}')
  if p.stat().st_size>5_000_000: raise ValueError(f'oversized:{rel}')
  if q.exists(): raise ValueError(f'destination_already_exists:{rel}')
  payload=p.read_bytes(); transformation='exact allowlisted bytes'
  if rel=='artifacts/postfreeze_mali_shots_ablation_v1/audit_preprocessing_v1/receipt.json':
   original=json.loads(payload)
   keys={'status','scope','checks_count','folds','global_input_rows_checked','checkpoint_replays',
         'source_shots','model_source_sha256','max_cpu_prediction_replay_difference',
         'sensitive_fixture_checkpoints','wall_seconds','cpu_threads','no_training','no_simulation'}
   payload=(json.dumps({k:v for k,v in original.items() if k in keys},indent=2)+'\n').encode()
   transformation='explicit aggregate receipt projection; private replay fixtures omitted'
  normalized=payload.decode().replace('\r\n','\n').replace('\r','\n').rstrip('\n')+'\n'
  if normalized.encode()!=payload:
   transformation+='; publication whitespace normalized to LF and one final newline'
   payload=normalized.encode()
  text=payload.decode()
  if p.suffix=='.csv':
   with p.open(newline='') as f: header=next(csv.reader(f))
   if FORBIDDEN_COLUMNS.intersection(header): raise ValueError(f'private_columns:{rel}')
  if p.suffix in {'.md','.csv','.json'} and re.search(r'(?:mali_real_qpu|qonductor_single_circuit_ibm|qpack_mcp)\|row\d+',text):
   raise ValueError(f'private_observation_example:{rel}')
  if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)',text):
   raise ValueError(f'credential_pattern:{rel}')
  if p.suffix=='.py': compile(text,str(p),'exec')
  payloads[rel]=payload
  records.append({'path':rel,'source_sha256':sha(p),'published_sha256':hashlib.sha256(payload).hexdigest(),
                  'bytes':len(payload),'transformation':transformation})
 for record in records:
  p=source/record['path']; q=dest/record['path']; q.parent.mkdir(parents=True,exist_ok=True)
  q.write_bytes(payloads[record['path']]); assert sha(q)==record['published_sha256']
 manifest={
  'scope':'aggregate_reports_and_source_only',
  'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
  'publication_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=dest,text=True).strip(),
  'private_qpu_rows_included':False,'training_or_measurement_executed_by_export':False,
  'excluded':['per-row/group QPU targets and predictions','QASM/graphs','checkpoints',
              'split/membership ledgers','bootstrap identity archives','raw logs',
              'unfinished random-row study'],
  'reproduction_limit':'Public verifier checks aggregates and hashes; original producers require retained private inputs and historical dependencies.',
  'files':records,
 }
 (dest/'docs/publication/RECENT_QPU_INDEX.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'status':'EXPORTED','files':len(records),'bytes':sum(r['bytes'] for r in records)}))

if __name__=='__main__': main()
