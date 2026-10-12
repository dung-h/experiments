#!/usr/bin/env python3
"""Read-only checks of public reporting and own Dense evidence; no model imports."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]

def rows(path):
 with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def close(a,b):
 if not math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12):raise ValueError(f'value differs:{a} vs {b}')

def verify(root=ROOT):
 root=Path(root)
 receipt=json.loads((root/'provenance/completed_science.json').read_text())
 if receipt['third_party_qpu_row_level_derivatives_included'] or receipt['source_QASM_or_upstream_assets_included']:
  raise ValueError('private publication scope')
 for item in receipt['files']:
  p=root/item['public_path']
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['published_sha256']:
   raise ValueError(f'export payload hash mismatch:{item["public_path"]}')
 table=root/'results/thesis/tables'
 rq=rows(table/'g_rq_closure.csv')
 if len(rq)!=10 or len({(r['rq'],r['track']) for r in rq})!=10:raise ValueError('RQ coverage')
 allowed={'ANSWERED','ANSWERED WITH QUALIFICATIONS','PARTIALLY ANSWERED','UNANSWERED'}
 if any(r['status'] not in allowed or r['blocking']!='NON-BLOCKING' for r in rq):raise ValueError('RQ closure status')
 closure={(r['rq'],r['track']):(r['status'],r['blocking']) for r in rows(root/'results/research_questions/closure/final_rq_status.csv')}
 if closure!={(r['rq'],r['track']):(r['status'],r['blocking']) for r in rq}:raise ValueError('closure/synthesis disagreement')
 main=rows(table/'b_qpu_main_accuracy.csv')
 if len(main)!=11 or any(int(r['n'])!=4515 for r in main):raise ValueError('QPU denominator')
 source={r['method_id']:r for r in rows(root/'results/analysis/qpu_metrics.csv')}
 for r in main:
  original=source[r['method']]
  for new,old in [('mae_s','mae_seconds'),('r2','r2_seconds'),('source_balanced_mae_s','source_balanced_mae'),('equal_group_mae_s','equal_group_mae')]:close(r[new],original[old])
 cost=rows(table/'e_accuracy_cost.csv')
 if len(cost)!=6:raise ValueError('cost cohort count')
 for r in cost:
  if r['context']=='QPU exact45':
   d=json.loads((root/'results/accuracy_cost/qpu/analysis.json').read_text())
   s=next(x for x in d['summaries'] if x['method']==r['method'] and x['source']=='all_selected');scale=1
   if int(r['n'])!=45 or r['accuracy_unit']!='s':raise ValueError('QPU cost unit')
  else:
   d=json.loads((root/'results/accuracy_cost/dense/analysis.json').read_text())
   precision='fp32' if 'FP32' in r['context'] else 'fp64'
   s=next(x for x in d['contexts'][precision]['summary'] if x['method']==r['method']);scale=1000
   if int(r['n'])!=30 or r['accuracy_unit']!='ms':raise ValueError('Dense cost unit')
  close(r['mae'],s['mae_seconds']*scale);close(r['rmse'],s['rmse_seconds']*scale)
  close(r['r2'],s['r2']);close(r['median_B2_ms'],s['median_request_median_B2_ms'])
 # Independently derive new Dense accuracy and B2 medians from published own records.
 pairs=rows(root/'results/accuracy_cost/dense/paired_circuit_results.csv')
 timings=rows(root/'results/accuracy_cost/dense/raw_b2_timings.csv')
 if len(pairs)!=60 or len(timings)!=3600:raise ValueError('Dense observation/repetition denominator')
 for context in ('dense_fp32','dense_fp64'):
  ps=[r for r in pairs if r['context']==context]
  if len(ps)!=30 or len({r['id'] for r in ps})!=30:raise ValueError('Dense unique circuit denominator')
  for method in ('graph','mlp'):
   output=next(r for r in cost if context.split('_')[1].upper() in r['context'] and r['method']==method)
   close(output['mae'],statistics.mean(abs(float(r['target_seconds'])-float(r[method+'_prediction_seconds'])) for r in ps)*1000)
   medians=[]
   for circuit in ps:
    values=[int(r['total_B2_ns']) for r in timings if r['context']==context and r['id']==circuit['id'] and r['method']==('mali_style_dense_graph' if method=='graph' else 'matched_metadata_mlp')]
    if len(values)!=30:raise ValueError('Dense repeat count')
    medians.append(statistics.median(values)/1e6)
   close(output['median_B2_ms'],statistics.median(medians))
 return {'status':'PASS','RQ_track_rows':10,'QPU_aggregate_methods':11,'matched_accuracy_cost_rows':6,'Dense_pairs':60,'Dense_repeats':3600,
         'public_payload_records':len(receipt['files']),'reproduction_scope':'saved reports/own Dense records only; QPU raw evidence excluded'}

if __name__=='__main__':print(json.dumps(verify(),indent=2))
