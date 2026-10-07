#!/usr/bin/env python3
"""Freeze the shared logical-eligible QPU panel and grouped splits without fitting."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'artifacts/real_qpu/logical_inputs'
OUTPUT = ROOT / 'artifacts/real_qpu/common_panel'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write(name, rows):
    with (OUTPUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit('Output is nonempty; preserve the frozen panel.')
    features = read(INPUT / 'feature_attempts.csv')
    old_outer = read(INPUT / 'logical_outer_splits.csv')
    groups = {r['canonical_observation_id']: r['group_id'] for r in old_outer}
    panel = [{key: r[key] for key in (
        'canonical_observation_id', 'source_id', 'logical_input_tier',
        'logical_input_digest', 'graph_file', 'graph_sha256', 'snapshot_id',
        'properties_sha256')} | {'group_id': groups[r['canonical_observation_id']]}
        for r in features if r['status'] == 'available']
    ids = {r['canonical_observation_id'] for r in panel}
    assert len(panel) == len(ids) == 7350
    sources = Counter(r['source_id'] for r in panel)
    assert sources == {'mali_real_qpu': 340, 'qonductor_single_circuit_ibm': 3065, 'qpack_mcp': 3945}
    source = {r['canonical_observation_id']: r['source_id'] for r in panel}
    members = defaultdict(list)
    for r in panel:
        members[r['group_id']].append(r['canonical_observation_id'])

    def assign(components, folds, seed):
        totals = Counter(source[rid] for rows in components.values() for rid in rows)
        loads = [Counter() for _ in range(folds)]
        result = {}
        ordered = sorted(components, key=lambda g: (
            -max(Counter(source[rid] for rid in components[g])[s] / totals[s] for s in totals),
            -len(components[g]), g))
        for group in ordered:
            vector = Counter(source[rid] for rid in components[group])
            def score(fold):
                cost = sum(((loads[f][s] + (vector[s] if f == fold else 0) - totals[s] / folds)
                            / max(totals[s] / folds, 1)) ** 2
                           for f in range(folds) for s in totals)
                tie = hashlib.sha256(f'{seed}:{group}:{fold}'.encode()).hexdigest()
                return cost, tie
            chosen = min(range(folds), key=score)
            result[group] = chosen
            loads[chosen].update(vector)
        return result, [dict(x) for x in loads]

    assignment, outer_counts = assign(members, 5, 42)
    outer = [{'canonical_observation_id': r['canonical_observation_id'],
              'group_id': r['group_id'], 'outer_fold': assignment[r['group_id']]} for r in panel]
    inner, inner_counts = [], {}
    for fold in range(5):
        train = {g: rows for g, rows in members.items() if assignment[g] != fold}
        inner_assignment, counts = assign(train, 4, 43 + fold)
        inner_counts[str(fold)] = counts
        assert all(set(count) == set(sources) for count in counts)
        for group, rows in train.items():
            inner.extend({'canonical_observation_id': rid, 'group_id': group,
                          'outer_fold': fold, 'inner_fold': inner_assignment[group]} for rid in rows)
    assert all(set(count) == set(sources) for count in outer_counts)
    for fold in range(5):
        test = {r['canonical_observation_id'] for r in outer if r['outer_fold'] == fold}
        rows = [r for r in inner if r['outer_fold'] == fold]
        assert len(rows) == len(ids - test)
        assert {r['canonical_observation_id'] for r in rows} == ids - test
        for key in ('group_id',):
            assert not ({r[key] for r in outer if r['outer_fold'] == fold}
                        & {r[key] for r in rows})
            for val in range(4):
                assert not ({r[key] for r in rows if r['inner_fold'] == val}
                            & {r[key] for r in rows if r['inner_fold'] != val})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write('panel.csv', panel)
    write('outer_splits.csv', outer)
    write('inner_splits.csv', inner)
    manifest = {'status': 'scientific_design_locked_execution_pending', 'rows': len(panel),
                'source_counts': dict(sources), 'tier_counts': dict(Counter(r['logical_input_tier'] for r in panel)),
                'outer_folds': 5, 'inner_folds': 4, 'validation_inner_fold': 0,
                'split_seed': 42, 'neural_seeds': [42, 1234, 31415],
                'groups': len(members), 'outer_source_counts': outer_counts,
                'inner_source_counts': inner_counts, 'group_leakage': False,
                'selection_and_split_read_target_labels': False,
                'selection': 'status=available in pinned logical feature ledger; no runtime/error selection',
                'group_rule': 'Retain full 8767 transitive component IDs, including excluded bridge rows; reassign folds using only panel source counts.',
                'assignment': 'Greedy normalized source-count balance, largest relative groups first, SHA256 seed/group/fold tie break.',
                'training_authorized': False,
                'input_hashes': {str(p.relative_to(ROOT)): sha(p) for p in (
                    INPUT / 'feature_attempts.csv', INPUT / 'logical_outer_splits.csv',
                    INPUT / 'model_input_groups.csv', INPUT / 'feature_manifest.json',
                    ROOT / 'benchmark_v1/scripts/freeze_real_qpu_panel.py')},
                'output_hashes': {name: sha(OUTPUT / name) for name in ('panel.csv', 'outer_splits.csv', 'inner_splits.csv')}}
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'rows': len(panel), 'groups': len(members), 'outer_source_counts': outer_counts, 'leakage': False}))


if __name__ == '__main__':
    main()
