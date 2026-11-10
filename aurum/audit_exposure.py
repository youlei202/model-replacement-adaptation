"""Audit historical rich-document membership, without reading test outcomes.

The remaining population is an eligibility inventory, not a selected new test.
A confirmation sample must only be selected after the AURUM method is frozen.
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

from .data import ROOT, LEGACY_ROOT, file_sha256


def main():
    out = ROOT / 'runs/aurum/audit'; out.mkdir(exist_ok=True, parents=True)
    columns = ['source', 'source_key', 'document_sha256', 'cluster_id', 'item_id', 'role']
    documents = pd.read_parquet(LEGACY_ROOT / 'documents.parquet', columns=columns)
    lockpath = LEGACY_ROOT / 'selection_lock.json'
    original_test = set(json.loads(lockpath.read_text())['test_item_ids'])
    supplement_path = LEGACY_ROOT / 'matched_resource_supplement/cohort.parquet'
    supplement = pd.read_parquet(supplement_path, columns=columns)
    memberships = {
        'legacy_pilot': set(documents.loc[documents.role.eq('pilot'), 'item_id']),
        'legacy_fit': set(documents.loc[documents.role.eq('fit'), 'item_id']),
        'legacy_validation': set(documents.loc[documents.role.eq('validation'), 'item_id']),
        'legacy_original_final_test': original_test,
        'legacy_matched_resource_supplement_final_test': set(supplement.item_id),
    }
    exposure = []
    union = set()
    for name, ids in memberships.items():
        if union & ids:
            raise ValueError(f'Unexpected historical membership overlap: {name}')
        if not ids <= set(documents.item_id):
            raise ValueError(f'Historical membership absent from locked document universe: {name}')
        union |= ids
        members = documents[documents.item_id.isin(ids)]
        exposure.append({'cohort': name, 'unique_documents': len(ids),
                         'source_counts': members.groupby('source').size().to_dict(),
                         'disposition': 'exclude_from_new_independent_confirmation'})
    remaining = documents[~documents.item_id.isin(union)].copy()
    if not remaining.role.eq('test_pool').all() or remaining.cluster_id.duplicated().any():
        raise ValueError('Unexpected remaining population membership')
    remaining.to_parquet(out / 'unexposed_candidates_metadata.parquet', index=False)
    cross = []
    for path in [LEGACY_ROOT / 'concurrent_mcq_cross_audit.json',
                 LEGACY_ROOT / 'matched_resource_supplement/concurrent_new_mcq_cross_audit.json']:
        audit = json.loads(path.read_text())
        cross.append({'path': str(path), 'sha256': file_sha256(path), 'passed': audit['passed'],
                      'cross_collision_count': audit['cross_collision_count'],
                      'scope': audit.get('known_scope', audit.get('scope', 'See immutable audit provenance'))})
    report = {'kind': 'metadata-only exposure inventory; not new confirmation collection',
              'historical_test_outcomes_read': False, 'historical_test_response_arrays_read': False,
              'known_population_documents': len(documents), 'excluded_exposed_documents': len(union),
              'cohorts': exposure, 'remaining_unexposed_documents': len(remaining),
              'remaining_source_counts': remaining.groupby('source').size().to_dict(),
              'max_balanced_confirmation_documents': int(2 * remaining.groupby('source').size().min()),
              'remaining_candidates_sha256': file_sha256(out / 'unexposed_candidates_metadata.parquet'),
              'selection_status': 'No AURUM confirmation rows selected; freeze method before deterministic selection.',
              'past_cross_population_dedup_audits': cross,
              'membership_artifacts': [{'path': str(p), 'sha256': file_sha256(p)} for p in
                                       [LEGACY_ROOT / 'documents.parquet', lockpath, supplement_path]],
              'scope_limit': 'Known repository campaigns and locked rich universe; recheck exposure if another campaign collects rows.'}
    (out / 'historical_exposure_inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['known_population_documents', 'excluded_exposed_documents',
                                           'remaining_unexposed_documents', 'remaining_source_counts',
                                           'max_balanced_confirmation_documents']}, indent=2))


if __name__ == '__main__':
    main()
