"""Saved-evidence closure checks; no fitting, inference or workloads."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('rq_closure', ROOT/'benchmark_v1/scripts/build_research_question_closure.py')
closure=importlib.util.module_from_spec(spec)
spec.loader.exec_module(closure)


def test_working_rqs_not_historical_numbering():
    assert len(closure.CARDS)==10
    assert {(c['rq'],c['track']) for c in closure.CARDS}=={(f'RQ{i}',t) for i in range(1,6) for t in 'AB'}
    assert all(c['status']=='PARTIALLY ANSWERED' for c in closure.CARDS if c['rq'] in ('RQ2','RQ4'))
    assert 'không phải old RQ4 grid' in closure.framework()


def test_context_coverage_does_not_promote_analog_or_portability():
    rows=closure.coverage()
    assert len(rows)==55
    assert all(r['status']=='UNANSWERED' for r in rows if r['context_id']=='analog' and r['rq'] in ('RQ1','RQ2'))
    assert all(r['status']=='PARTIALLY ANSWERED' for r in rows if r['rq']=='RQ4')
    assert all(r['track']=='B' for r in rows if r['context_id']=='historical_width')


def test_failures_answer_questions_and_cost_requires_evidence():
    assert all(c['status']=='ANSWERED WITH QUALIFICATIONS' for c in closure.CARDS if c['rq']=='RQ3')
    gaps=[g for g in closure.GAPS if g[1]=='G2']
    assert len(gaps)==1 and gaps[0][0]=='G01'
    assert 'NOT AUTHORIZED' in closure.AMENDMENT
    assert 'không neural retrain' in closure.AMENDMENT


def test_numerical_provenance_and_timer_boundaries():
    f=closure.facts()
    assert f['claims']['C01']['graph_mae']<f['claims']['C01']['median_mae']
    assert f['claims']['C02']['ci_low']<0<f['claims']['C02']['ci_high']
    assert all(v['cells']==15 for v in f['training_receipts'].values())
    assert f['tn_ratios']['frontier_q10_q16']>f['tn_ratios']['core_q2_q9']
    assert f['tn_phases']['t_path_search_s']['n']==603
    assert f['tn_phases']['end_to_end_nonempty_rows']==0


def test_saved_reports_replay_and_frozen_sources_match():
    result=closure.validate()
    assert result['status']=='PASS'
    assert result['scientific_verdict']=='PARTIALLY SUFFICIENT'
    assert result['amendments_authorized']==0
    assert result['no_training_inference_simulation_QCEC'] is True
