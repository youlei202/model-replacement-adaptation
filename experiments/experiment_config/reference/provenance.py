"""Scientific workflow checks, not a security sandbox against a malicious agent."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def assert_fresh(new: Iterable[Mapping], historical: Iterable[Mapping]) -> None:
    """Requires global dedup IDs already constructed across old AND new data.
    Does not itself perform near-duplicate clustering.
    """
    columns = ('global_cluster_id', 'normalized_text_sha256')
    old = list(historical)
    fresh = list(new)
    if not old or not fresh:
        raise ValueError('Nonempty historical registry and new cohort required')
    for rows in (old, fresh):
        if any(any(not r.get(c) for c in columns) for r in rows):
            raise ValueError('Global cluster and normalized-text digests are mandatory')
    for col in columns:
        overlap = {r[col] for r in fresh} & {r[col] for r in old}
        if overlap:
            raise ValueError(f'Historical exposure overlap via {col}: {len(overlap)}')


def assert_split_disjoint(partitions: Mapping[str, Iterable[Mapping]]) -> None:
    seen = {}
    for role, rows in partitions.items():
        for row in rows:
            cluster = row.get('global_cluster_id')
            if not cluster:
                raise ValueError('Missing global cluster ID')
            if cluster in seen and seen[cluster] != role:
                raise ValueError(f'Cluster spans roles: {seen[cluster]} and {role}')
            seen[cluster] = role


def freeze_artifacts(root: Path, paths: Iterable[Path], metadata: dict) -> dict:
    root = Path(root).resolve()
    artifacts = {}
    for path in paths:
        path = Path(path).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError('Freeze only existing files within the run directory')
        artifacts[str(path.relative_to(root))] = sha256(path)
    if not artifacts:
        raise ValueError('Empty freeze is not a valid experiment lock')
    value = {'metadata': metadata, 'artifacts': artifacts}
    value['lock_digest'] = canonical_digest(value)
    return value


def verify_freeze(root: Path, lock: Mapping) -> None:
    root = Path(root).resolve()
    body = {k: v for k, v in lock.items() if k != 'lock_digest'}
    if canonical_digest(body) != lock.get('lock_digest'):
        raise ValueError('Freeze metadata was modified')
    if not body.get('artifacts'):
        raise ValueError('Empty artifact lock')
    for rel, expected in body['artifacts'].items():
        path = (root / rel).resolve()
        if root not in path.parents or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Frozen artifact missing or modified: {rel}')


def require_confirmation_metadata(metadata: Mapping) -> None:
    if metadata.get('data_role') != 'fresh_confirmation':
        raise ValueError('Discovery or reseeded historical data is not fresh confirmation')
    for field in ('historical_exposure_audit_passed', 'global_near_duplicate_audit_passed',
                  'selected_before_test', 'complete_candidate_family',
                  'strong_baselines_complete'):
        if metadata.get(field) is not True:
            raise ValueError(f'Confirmation requirement not satisfied: {field}')
    if int(metadata.get('independent_test_clusters', 0)) < 2:
        raise ValueError('Insufficient independent test clusters')
