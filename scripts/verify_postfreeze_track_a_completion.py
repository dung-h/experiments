#!/usr/bin/env python3
"""Verify the public report package; requires no private QPU rows or models."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline='') as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    index = json.loads((ROOT / 'docs/final_completion/PUBLICATION_INDEX.json').read_text())
    assert index['scope'] == 'aggregate_reports_and_code_only'
    for entry in index['files']:
        path = ROOT / entry['path']
        assert path.is_file() and not path.is_symlink(), entry['path']
        assert path.stat().st_size == entry['bytes'], entry['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], entry['path']
    sample = 'artifacts/postfreeze_track_a_transfer_sample_efficiency_v1/aggregation_v2/'
    perf = {(r['fraction'], r['treatment']): r for r in read(sample + 'model_results.csv')}
    assert len(perf) == 16
    for row in read(sample + 'paired_comparisons.csv'):
        left, right = row['contrast'].split(' - ')
        expected = float(perf[(row['fraction'], left)]['group_balanced_mae']) - float(perf[(row['fraction'], right)]['group_balanced_mae'])
        assert math.isclose(expected, float(row['delta_group_mae']), abs_tol=1e-10)
    completion = json.loads((ROOT / (sample + 'completion_receipt.json')).read_text())
    assert completion['status'] == 'PASS_FROZEN' and completion['training_cells'] == 360
    assert completion['parent_100_unchanged'] and completion['replay_tests_pass']
    candidates = {'node_only': .6930083261, 'true_dag': .6550087318,
                  'reversed_dag': .6161665613, 'shuffled_dag': .6541041634}
    for row in read('artifacts/postfreeze_dag_raw_r2_audit_v1/dag_raw_metrics.csv'):
        if row['scope'] == 'pooled':
            assert int(row['n_rows']) == 340 and int(row['n_raw_qasm_groups']) == 170
            assert math.isclose(float(row['r2_raw_seconds']), candidates[row['variant']], abs_tol=1e-9)
    trans = 'artifacts/postfreeze_track_a_transpiled_representation_v1/p1_representation/'
    scores = {(r['learner'], r['view']): float(r['group_balanced_mae_seconds']) for r in read(trans + 'aggregate/pooled_metrics.csv')}
    assert len(scores) == 10
    for row in read(trans + 'aggregate/paired_comparisons.csv'):
        expected = scores[(row['learner'], row['left_view'])] - scores[(row['learner'], row['right_view'])]
        assert math.isclose(expected, float(row['point_group_balanced_mae_delta_seconds']), abs_tol=1e-10)
    for relative in ['research/postfreeze_track_a_transfer_sample_efficiency_v1',
                     'research/postfreeze_track_a_transpiled_representation_v1',
                     'research/postfreeze_dag_raw_r2_audit_v1']:
        for path in (ROOT / relative).glob('*.py'):
            compile(path.read_text(), str(path), 'exec')
    print(f'PASS: {len(index["files"])} byte-hashed report/code files; aggregate deltas, coverage metadata, raw-R2 candidates, Python syntax. No private-row replay or model execution.')


if __name__ == '__main__':
    main()
