#!/usr/bin/env python3
"""Fresh supplemental MCQ reservoir versus all rich full documents/contexts."""
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from experiments.confirmation_prepare import signature, shingles
from experiments.exposure_audit import text_hash
from rich_setup import OUT, digest, dump


def main():
    source = ROOT / 'experiments/runs/v2/resource_cap_confirmation/unused_candidate_reservoir.parquet'
    output = OUT / 'matched_resource_supplement/concurrent_new_mcq_cross_audit.json'
    if output.exists():
        old = json.load(open(output)); assert old['mcq_reservoir_sha256'] == digest(source)
        print('Existing immutable crossaudit', old['cross_collision_count']); return
    hist = pd.read_parquet(ROOT / 'experiments/runs/v2/audit/historical_text_registry.parquet')
    fresh = pd.read_parquet(OUT / 'candidate_documents.parquet')
    kept = pd.read_parquet(OUT / 'documents.parquet')
    roles = kept.set_index('item_id').role.to_dict(); texts = []; unique = {}; richnodes = {}
    def node(text):
        h = text_hash(text)
        if h not in unique:
            unique[h] = len(texts); texts.append(text)
        return unique[h]
    for r in hist.itertuples():
        node(r.context); node(r.context + '\n' + '\n'.join(json.loads(r.options)))
    for r in fresh.itertuples():
        a, b = node(r.text), node(r.context_text)
        if r.item_id in roles:
            richnodes.setdefault(a, []).append((r.item_id, roles[r.item_id]))
            richnodes.setdefault(b, []).append((r.item_id, roles[r.item_id]))
    raw = np.load(OUT / 'rich_duplicate_signatures.npz'); sig = raw['signatures']
    assert len(texts) == len(sig)
    assert str(raw['fingerprint']) == hashlib.sha256(json.dumps([text_hash(t) for t in texts]).encode()).hexdigest()
    buckets = defaultdict(list)
    for i in richnodes:
        for b in range(8):
            buckets[(b, sig[i, b*8:b*8+8].tobytes())].append(i)
    frame = pd.read_parquet(source); queries = []
    for r in frame.itertuples():
        queries.extend([(r.item_id, r.context), (r.item_id, r.context + '\n' + '\n'.join(json.loads(r.options)))])
    @lru_cache(256)
    def sh(i):
        return shingles(texts[i])
    edges = []
    with ProcessPoolExecutor(8) as ex:
        for (item, text), s in zip(queries, ex.map(signature, [r[1] for r in queries], chunksize=32)):
            arr = np.asarray(s, dtype=np.uint64); matches = set(); exact = unique.get(text_hash(text))
            if exact in richnodes:
                matches.add(exact)
            for b in range(8):
                matches.update(buckets.get((b, arr[b*8:b*8+8].tobytes()), []))
            a = shingles(text)
            for j in matches:
                b = sh(j); jac = len(a & b) / len(a | b) if a | b else 1
                if jac >= .85:
                    for richitem, role in richnodes[j]:
                        edges.append({'mcq_item_id': item, 'rich_item_id': richitem, 'rich_role': role, 'jaccard': jac})
    output.parent.mkdir(exist_ok=True)
    dump(output, {'complete': True, 'passed': not edges, 'new_outputs_collected': False,
                 'mcq_reservoir_path': str(source.relative_to(ROOT)), 'mcq_reservoir_sha256': digest(source),
                 'mcq_reservoir_item_ids': frame.item_id.tolist(), 'mcq_reservoir_rows': len(frame),
                 'rich_documents_sha256': digest(OUT / 'documents.parquet'), 'rich_kept_documents': len(kept),
                 'signature_sha256': digest(OUT / 'rich_duplicate_signatures.npz'),
                 'original_rule': 'normalized word5 shingles;64 BLAKE2b64 minhash;8bands8; exact verification Jaccard>=.85 plus exact normalized texts; full documents and actual contexts',
                 'cross_collisions': edges, 'cross_collision_count': len(edges), 'code_sha256': digest(__file__)})
    print('Crossaudit', len(frame), 'MCQ reservoir,', len(kept), 'rich full documents and contexts; collisions', len(edges), flush=True)


if __name__ == '__main__':
    main()
