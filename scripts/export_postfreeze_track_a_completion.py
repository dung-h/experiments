#!/usr/bin/env python3
"""Copy an explicit report/code allowlist into an isolated existing checkout.

Excluded local scientific evidence stays untouched. No Git operation, model
execution, simulator call, or recursive directory export is performed here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def selected_paths() -> list[str]:
    paths = [
        'scripts/export_postfreeze_track_a_completion.py',
        'scripts/verify_postfreeze_track_a_completion.py',
    ]
    groups = {
        'docs/final_completion': [
            'CLAIM_MATRIX.md', 'EXISTING_EVIDENCE_AUDIT.md', 'EXPERIMENT_STOP_GO.md',
            'FINAL_COMPLETION_REPORT.md', 'FINAL_RESULT_HIERARCHY.md',
            'FROZEN_EVALUATION_CONTRACT.md', 'FUTURE_CIRCUIT_QPU_ENCODER.md',
            'P1_REPRESENTATION_EXPLAINED.md', 'PHYSICAL_DAG_TRANSFORMER_STRATEGY_DRAFT.md',
            'PRESENTATION_UPDATE_BRIEF.md', 'WORK_STATUS_20261023.md', 'PUBLICATION_SCOPE.md',
        ],
        'docs/audits/qpu_timing_boundary': [
            'SOURCE_TARGET_SEMANTICS.md', 'ANALYTICAL_ROUTE_AUDIT.md',
            'ERROR_DECOMPOSITION_AUDIT.md', 'EVIDENCE_MATRIX.csv', 'FINAL_VERDICT.md',
        ],
        'research/postfreeze_track_a_transfer_sample_efficiency_v1': [
            'run_sample_efficiency.py', 'run_safe_cells.py', 'benchmark_batch_capacity.py',
            'aggregate_sample_efficiency.py', 'aggregate_validated_v2.py',
            'freeze_aggregation_v2.py', 'test_aggregation_v2.py',
        ],
        'artifacts/postfreeze_track_a_transfer_v1': ['run_qpu_factorial.py', 'run_strict_primary.py'],
        'artifacts/postfreeze_track_a_transfer_sample_efficiency_v1': ['AGGREGATION_SPEC_V2.md'],
        'artifacts/postfreeze_track_a_transfer_sample_efficiency_v1/aggregation_v2': [
            'CLAIM_MATRIX.md', 'P1_SAMPLE_EFFICIENCY_REPORT.md', 'model_results.csv',
            'paired_comparisons.csv', 'fold_metrics.csv', 'source_metrics.csv',
            'seed_metrics.csv', 'seed_paired_deltas.csv', 'stratified_paired_deltas.csv',
            'subset_metrics.csv', 'cell_validation.csv', 'completion_receipt.json',
            'replay_test_results.txt', 'absolute_performance.png',
            'heterogeneity.png', 'transfer_effect.png',
        ],
        'research/postfreeze_track_a_transpiled_representation_v1': [
            'run_p0_coverage_audit.py', 'prepare_p1_features.py',
            'run_p1_representation.py', 'aggregate_p1_representation.py',
            'validate_and_finalize_p1.py',
        ],
        'artifacts/postfreeze_track_a_transpiled_representation_v1': [
            'PROTOCOL.md', 'FINAL_REPORT.md', 'p1_manifest.json', 'p1_manifest.sha256',
            'p1_secondary_hgb_manifest.json',
        ],
        'artifacts/postfreeze_track_a_transpiled_representation_v1/features': ['feature_dictionary.json', 'feature_hashes.json'],
        'artifacts/postfreeze_track_a_transpiled_representation_v1/logical_topology_existing_evidence': ['REPORT.md'],
        'artifacts/postfreeze_track_a_transpiled_representation_v1/p0_coverage': [
            'REPORT.md', 'receipt.json', 'source_coverage.csv', 'backend_coverage.csv',
            'provenance_coverage.csv', 'dag_size_distribution.csv',
        ],
        'artifacts/postfreeze_track_a_transpiled_representation_v1/p1_representation': ['REPORT.md', 'validation_receipt.json'],
        'artifacts/postfreeze_track_a_transpiled_representation_v1/p1_representation/aggregate': [
            'receipt.json', 'pooled_metrics.csv', 'paired_comparisons.csv',
            'fold_metrics.csv', 'source_metrics.csv', 'provenance_metrics.csv',
        ],
        'artifacts/postfreeze_track_a_transpiled_representation_v1/p1_representation/ridge': [
            'receipt.json', 'feature_names.json', 'hyperparameter_receipts.csv',
        ],
        'artifacts/postfreeze_track_a_transpiled_representation_v1/p1_representation/hgb': [
            'receipt.json', 'feature_names.json', 'hyperparameter_receipts.csv',
        ],
        'research/postfreeze_dag_raw_r2_audit_v1': ['verify_dag_raw_r2.py', 'test_verify_dag_raw_r2.py'],
        'artifacts/postfreeze_dag_raw_r2_audit_v1': [
            'DAG_RAW_R2_VERIFICATION.md', 'DAG_TOPOLOGY_MECHANISM_AUDIT.md',
            'dag_raw_metrics.csv', 'dag_raw_paired_comparisons.csv', 'verification_receipt.json',
        ],
        'artifacts/postfreeze_final_completion_v1': ['README.md', 'INTEGRITY_RECEIPT.md', 'cost_audit.md', 'ood_support_diagnostic.csv'],
    }
    for folder, names in groups.items():
        paths.extend(f'{folder}/{name}' for name in names)
    return sorted(paths)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_selected(source: Path, paths: list[str]) -> None:
    forbidden_columns = {
        'canonical_observation_id', 'canonical_row_id', 'row_id', 'group_id',
        'raw_qasm_sha256', 'source_qasm_sha256', 'target_seconds',
        'actual_seconds', 'prediction_seconds',
    }
    if len(paths) != len(set(paths)):
        raise ValueError('duplicate_allowlist_path')
    for name in paths:
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'missing_or_symlink:{name}')
        if path.suffix == '.csv':
            with path.open(newline='') as handle:
                header = next(csv.reader(handle))
            if forbidden_columns.intersection(header):
                raise ValueError(f'row_level_column_in_export:{name}')
        if path.stat().st_size > 5_000_000:
            raise ValueError(f'oversized_selected_file:{name}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if source == destination or not (destination / '.git').exists():
        raise ValueError('destination_must_be_separate_existing_checkout')
    paths = selected_paths()
    validate_selected(source, paths)
    # Fail before any copying if a selected path would overwrite checkout data.
    for name in paths:
        if (destination / name).exists():
            raise ValueError(f'output_already_exists:{name}')
    entries = []
    for name in paths:
        origin, target = source / name, destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        assert sha(origin) == sha(target)
        entries.append({'path': name, 'sha256': sha(target), 'bytes': target.stat().st_size})
    index = {
        'scope': 'aggregate_reports_and_code_only',
        'source_scientific_head': '146886cff0d3f329291197be0d742b4743735c88',
        'remote_main_base': '39991eb48b82593ffc53914bb0136393af120588',
        'excluded': ['QPU observation rows', 'row/group predictions and errors',
                     'QASM and graphs', 'checkpoint weights', 'membership ledgers',
                     'group bootstrap archives', 'raw logs'],
        'files': entries,
    }
    output = destination / 'docs/final_completion/PUBLICATION_INDEX.json'
    output.write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({'status': 'COPIED', 'files': len(paths), 'bytes': sum(x['bytes'] for x in entries)}))


if __name__ == '__main__':
    main()
