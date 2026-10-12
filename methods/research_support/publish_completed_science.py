#!/usr/bin/env python3
"""Controlled public export of completed science; private evidence stays intact.

Export only reports, aggregate tables, project support code/tests and the project's
Dense measurements. Do not execute scientific producers or rewrite their sources.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PHASES={
 'benchmark_interpretation':'results/interpretation/benchmark',
 'thesis_interpretation':'results/interpretation/thesis',
 'research_question_closure':'results/research_questions/historical',
 'am1_feasibility':'results/accuracy_cost/feasibility',
 'am1_overhead_pilot':'results/accuracy_cost/pilot',
 'am1_postpilot_audit':'results/accuracy_cost/postpilot_audit',
 'am1_phase2':'results/accuracy_cost/pipeline_pilot',
 'am1_preprocessing_recovery':'results/reproducibility/preprocessing',
 'am1_phase4':'results/accuracy_cost/qpu',
 'am1_phase5_dense_b2':'results/accuracy_cost/dense',
 'final_exhaustive_validation':'results/validation/exhaustive',
 'final_rq_closure_v2':'results/research_questions/closure',
 'thesis_synthesis':'docs/thesis',
}
PRIVATE_COLUMNS={'id','observation_id','canonical_observation_id','target_seconds','actual_seconds',
 'saved_target_seconds','prediction_seconds','predicted_seconds','test_id','fit_id','test_target_seconds'}
SAFE_CSV={
 'analytical_predictive_skill.csv','benchmark_question_matrix.csv','effective_input_widths.csv',
 'generalization_decomposition.csv','generalization_field_support.csv','generalization_predictive_skill.csv',
 'global_feature_ledger.csv','masked_feature_concentration.csv','qpu_predictive_skill.csv',
 'qpu_source_balanced_skill.csv','shot_diversity.csv','simulator_predictive_skill.csv','track_a_fidelity.csv',
 'evidence_matrix_crosswalk.csv','evidence_to_claims.csv','analytical_controls.csv','estimator_identity.csv',
 'feature_structural_contribution.csv','generalization_robustness.csv','method_context_index.csv','open_questions.csv',
 'track_a_performance.csv','track_b_findings.csv','rq_coverage_matrix.csv','estimator_inventory.csv',
 'accuracy_cost_comparisons.csv','paired_latency.csv','selected_variants.csv','timing_summary.csv',
 'accuracy_population_metrics.csv','replay_denominators.csv','simulator_comparisons.csv',
 'fitted_state_recovery_inventory.csv','dense_replay_validation.csv','accuracy_latency_summary.csv',
 'stage_timing_summary.csv','full_context_accuracy.csv','final_rq_evidence_index.csv',
 'final_claim_matrix.csv','final_rq_status.csv','historical_hash_mismatches.csv','pathless_hash_resolution.csv',
}
SUPPORT_NAMES={
 'am1_preprocessing_compat.py','analyze_am1_phase2.py','analyze_am1_phase4.py','analyze_am1_phase5_dense_b2.py',
 'build_am1_feasibility.py','build_am1_postpilot_audit.py','build_final_rq_closure_v2.py',
 'build_research_question_closure.py','build_thesis_interpretation.py','build_thesis_synthesis.py',
 'export_thesis_tables.mjs','interpret_frozen_benchmark.py','recover_am1_preprocessing.py',
 'report_am1_phase4.py','report_am1_preprocessing_recovery.py','run_am1_overhead_pilot.py',
 'run_am1_phase2.py','run_am1_phase4.py','run_am1_phase5_dense_b2.py','summarize_am1_overhead_pilot.py',
 'validate_am1_overhead_pilot.py','validate_am1_phase5_dense_b2.py','validate_benchmark_interpretation.py',
}
ANALYSIS_TABLES={
 'qpu_metrics_v1.csv','qpu_source_metrics_v1.csv','qpu_source_paired_v1.csv','qpu_shared_reference_comparisons_v1.csv',
 'qpu_source_balanced_paired_v2.csv','analytical_controls_v1.csv','analytical_shared_comparisons_v1.csv',
 'simulator_metrics_v1.csv','simulator_paired_v1.csv','mps_selection_safety_v1.csv','tn_same_plan_slices_v1.csv',
 'pasqal_stage_quality_v1.csv','generalization_distributions_v1.csv','generalization_case_coverage_v1.csv',
 'generalization_seed_metrics_v1.csv','generalization_feature_support_v1.csv','kyoto_consumed_input_collisions_v1.csv',
}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

def semantic(name):
 return re.sub(r'_v\d+(?=\.|$)','',name)

def group(phase):
 if phase in {'benchmark_interpretation','thesis_interpretation','research_question_closure'}:return 1
 if phase in {'am1_feasibility','am1_overhead_pilot','am1_postpilot_audit','am1_phase2','am1_phase4','am1_phase5_dense_b2'}:return 2
 if phase=='am1_preprocessing_recovery':return 3
 return 4

def get_candidates():
 records=[]
 for phase,public in PHASES.items():
  for source in sorted((ROOT/'artifacts'/phase).rglob('*')):
   if not source.is_file():continue
   relative=source.relative_to(ROOT/'artifacts'/phase)
   parts=list(relative.parts)
   if parts[0]=='v1':parts=parts[1:]
   parts=[semantic(p) for p in parts]
   dest=public+'/'+('/'.join(parts))
   if phase=='thesis_synthesis' and parts[0]=='canonical_tables':dest='results/thesis/tables/'+'/'.join(parts[1:])
   suffix=source.suffix
   keep=False;reason='Private/mixed row-level evidence, redundant raw registry, worker output or historical metadata; retained locally'
   if source.is_symlink():reason='Symlink not allowed'
   elif suffix=='.md':keep=True;reason='Scientific report/qualification/validation receipt; exported with source-lineage and link adaptation'
   elif suffix=='.csv':
    if phase=='thesis_synthesis' or source.name in SAFE_CSV:keep=True;reason='Reviewed aggregate/lineage table, not row-level QPU targets or predictions'
    if phase=='am1_phase5_dense_b2' and source.name in {'paired_circuit_results.csv','raw_b2_timings.csv','replay_validation.csv'}:
     keep=True;reason='Project-measured simulator timing/prediction/replay evidence; no source QASM bytes'
    if keep and phase!='am1_phase5_dense_b2':
     with source.open(errors='strict') as f:fields=next(csv.reader(f))
     if PRIVATE_COLUMNS.intersection(fields):keep=False;reason='Private/mixed observation-level QPU columns; original retained locally'
   if keep and source.stat().st_size>5*1024*1024:raise ValueError(f'Unexpected large selected file:{source}')
   records.append(dict(source_path=source.relative_to(ROOT).as_posix(),source_sha256=sha(source),bytes=source.stat().st_size,
    disposition='INCLUDE' if keep else 'EXCLUDE',reason=reason,public_path=dest if keep else None,commit_group=group(phase)))
 # Exact project-owned producers and regression tests, not upstream code/checkpoints.
 for source in sorted((ROOT/'benchmark_v1/scripts').iterdir()):
  if source.name not in SUPPORT_NAMES:continue
  g=3 if 'preprocessing' in source.name else 4 if any(k in source.name for k in ('closure','synthesis')) else 1 if any(k in source.name for k in ('interpret','research_question')) else 2
  records.append(dict(source_path=source.relative_to(ROOT).as_posix(),source_sha256=sha(source),bytes=source.stat().st_size,
   disposition='INCLUDE',reason='Exact historical project producing/reporting source; archived, external/private prerequisites apply',
   public_path='methods/research_support/'+semantic(source.name),commit_group=g))
 for source in sorted((ROOT/'benchmark_v1/tests').glob('test_*.py')):
  if not any(k in source.name for k in ('am1','benchmark_interpretation','thesis_interpretation','research_question_closure')):continue
  records.append(dict(source_path=source.relative_to(ROOT).as_posix(),source_sha256=sha(source),bytes=source.stat().st_size,
   disposition='INCLUDE',reason='Exact historical regression-test source; full local fixtures needed, not default public unittest suite',
   public_path='methods/research_support/tests/'+source.name,commit_group=3 if 'preprocessing' in source.name else 4 if any(k in source.name for k in ('closure','thesis')) else 2))
 for source in sorted((ROOT/'artifacts/benchmark_deep_dive/tables').glob('*.csv')):
  if source.name not in ANALYSIS_TABLES:continue
  with source.open() as f:fields=next(csv.reader(f))
  assert not PRIVATE_COLUMNS.intersection(fields),source
  records.append(dict(source_path=source.relative_to(ROOT).as_posix(),source_sha256=sha(source),bytes=source.stat().st_size,
   disposition='INCLUDE',reason='Aggregate scientific supporting table, source hashes/units retained',
   public_path='results/analysis/'+semantic(source.name),commit_group=1))
 for src,dst in [('artifacts/final_research/rq4_report.md','results/generalization/report.md'),
                 ('artifacts/real_qpu/provenance_equivalence/report.md','results/reproducibility/qcec_report.md')]:
  p=ROOT/src
  records.append(dict(source_path=src,source_sha256=sha(p),bytes=p.stat().st_size,disposition='INCLUDE',
   reason='Final scientific report, no source circuit bytes/row-level target export',public_path=dst,commit_group=1))
 source=Path(__file__).resolve()
 records.append(dict(source_path=source.relative_to(ROOT).as_posix(),source_sha256=sha(source),bytes=source.stat().st_size,
  disposition='INCLUDE',reason='Controlled export selection source; documented private/development prerequisites',
  public_path='methods/research_support/publish_completed_science.py',commit_group=4))
 return records

def scan_secrets(payload,name):
 patterns=[rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'AKIA[0-9A-Z]{16}',rb'sk-(?:proj-)?[A-Za-z0-9_-]{40,}']
 if any(re.search(pattern,payload) for pattern in patterns):raise ValueError(f'Potential credential in {name}; stopped without printing content')

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('destination',type=Path);args=ap.parse_args()
 dest=args.destination.resolve();assert dest==(ROOT/'work/completed-science-publication').resolve()
 before=subprocess.check_output(['git','rev-parse','HEAD'],cwd=dest,text=True).strip()
 records=get_candidates();included=[r for r in records if r['disposition']=='INCLUDE']
 mapping={r['source_path']:r['public_path'] for r in included}
 assert len(set(mapping.values()))==len(mapping),'public filename collision'
 # Local inventory remains outside the public tree: every specified phase file is classified.
 local=ROOT/'work/completed-science-review';local.mkdir(exist_ok=True)
 (local/'file_decisions.json').write_text(json.dumps(records,indent=2)+'\n')
 changes=[]
 for row in included:
  source=ROOT/row['source_path']; target=dest/row['public_path'];payload=source.read_bytes()
  assert sha(source)==row['source_sha256']
  scan_secrets(payload,row['source_path'])
  transform='exact bytes'
  if target.suffix=='.md':
   original=payload.decode();old_dir=source.parent
   def replace(match):
    label,raw=match.groups();link=raw.strip('<>').split('#',1)[0]
    if raw.startswith(('http:','https:','mailto:','#')):return match.group(0)
    resolved=(old_dir/link).resolve()
    try:key=resolved.relative_to(ROOT).as_posix()
    except ValueError:key=''
    public=mapping.get(key)
    if public:
     import os
     return '['+label+']('+os.path.relpath(dest/public,target.parent)+')'
    return label+' (historical source; not bundled)'
   text=re.sub(r'\[([^\]\n]+)\]\(([^)\n]+)\)',replace,original)
   # Private per-observation examples can occur in prose as well as CSVs.
   chunks=text.split('\n\n')
   redacted=0
   for i,chunk in enumerate(chunks):
    if re.search(r'(?:mali_real_qpu|qonductor_single_circuit_ibm|qpack_mcp)\|row\d+',chunk):
     chunks[i]='Per-observation private QPU example omitted from this publication view; original evidence is retained locally.'
     redacted+=1
   text='\n\n'.join(chunks)
   text='> Publication view of saved scientific evidence. Historical paths, hashes and execution restrictions describe the producing snapshot, not new execution authority. Current closure: [scientific synthesis]('+__import__('os').path.relpath(dest/'docs/thesis/README.md',target.parent)+').\n\n'+text
   payload=text.encode();transform=f'Markdown links retargeted; absent/private evidence links explicitly marked; {redacted} private-example blocks omitted; retained aggregate numbers unchanged'
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
  changes.append({**row,'published_sha256':sha(target),'transformation':transform})
 # Export two aggregate analysis objects, deliberately excluding private observation/checkpoint fields.
 for phase,keys in [('am1_phase4',{'status','summaries','shares','paired','uncertainty','source_counts','source_balanced_macro_mae_seconds','correlations_descriptive_only'}),
                    ('am1_phase5_dense_b2',{'status','contexts','shares','full_context_accuracy'})]:
  src=ROOT/'artifacts'/phase/'v1/analysis.json';d=json.loads(src.read_text());result={k:d[k] for k in sorted(keys) if k in d}
  public=PHASES[phase]+'/analysis.json';p=dest/public;p.write_text(json.dumps(result,indent=2)+'\n')
  changes.append(dict(source_path=str(src.relative_to(ROOT)),source_sha256=sha(src),public_path=public,published_sha256=sha(p),
   disposition='INCLUDE_DERIVED_VIEW',reason='Aggregate-only object projection; no retained field value changed',commit_group=2,transformation='explicit aggregate key selection'))
 # Keep support code byte-exact. Public checking does not run private-data experiments.
 p=dest/'methods/research_support/README.md'
 p.write_text('''# Historical research support source

These project source files and regression tests are exact copies of the local
producing/reporting implementations. File names are semantic; original paths,
hashes and transformations are recorded in `provenance/completed_science.json`.
Private QPU labels, tensors, checkpoints and frozen row partitions are not
included. Some imports and path constants reference the original development
layout. These are source-audit records, not portable public refit commands.
Do not invoke measurement/inference runners as publication verification.

The public checks operate on published aggregate tables and the project's own
Dense timing/prediction records. Full original pytest fixtures and upstream
environments are required for the archived tests. No new training or timing is
needed to verify the public reporting payload. Framework/dependency and upstream
license attribution remain in the existing project notices; no blanket MIT
license or unverified author metadata is added by this export.
''')
 mapping_public={r['public_path']:r for r in changes}
 manifest=dict(scope='Completed scientific reports, aggregate QPU evidence, own simulator measurements and project research-support source',
  base_remote_commit=before,source_development_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  experiments_closed=True,third_party_qpu_row_level_derivatives_included=False,source_QASM_or_upstream_assets_included=False,
  checkpoints_or_environments_included=False,license_metadata_not_invented=True,
  files=changes,reviewed_phase_files=sum(r['source_path'].startswith('artifacts/'+p+'/') for r in records for p in PHASES),
  excluded_phase_file_count=sum(r['disposition']=='EXCLUDE' for r in records),
  excluded_categories=['Private/mixed QPU rows','Raw/redundant exhaustive indexes','Worker/process outputs','Predeclarations/input paths with private fixtures',
  'Redundant per-precision timing files','Environments/checkpoints/source QASM','Unrelated older dirty-worktree edits'],
  scientific_verdict='SCIENTIFICALLY SUFFICIENT WITH MATERIAL LIMITATIONS',
  public_reproduction_scope='Saved aggregate/own-Dense reporting validation; not self-contained QPU refit or historical model inference')
 (dest/'provenance/completed_science.json').write_text(json.dumps(manifest,indent=2)+'\n')
 # Baseline publication file inventory is refreshed after the new public tests/docs are finalized.
 assert all(sha(ROOT/r['source_path'])==r['source_sha256'] for r in records)
 print(json.dumps({'reviewed':len(records),'selected':len(changes),'excluded':sum(r['disposition']=='EXCLUDE' for r in records),
                  'source_hashes_unchanged':True,'group_file_counts':{g:sum(r['commit_group']==g for r in changes) for g in range(1,5)}},indent=2))

if __name__=='__main__':main()
