#!/usr/bin/env python3
"""Additional conservative MCQ probability-endpoint check; no scientific edits.

The original coordinate timings and their frozen flags remain intact. This layer
can only remove a positive operational claim; it never changes fitted methods,
query cohorts, cap sets, losses, or original multiplicity adjustments.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/runs/v2'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def truth(value):return str(value).lower() in ('true','1','1.0')


def register(run):
    path=run/'protocol/probability_endpoint_report_guard.json'
    if path.exists():
        assert json.loads(path.read_text())['analysis_script_sha256']==sha(__file__)
        return
    base=run/'resource_cap_confirmation/probability_endpoint_audit'
    dump(path,{'recorded_unix':time.time(),'analysis_script_sha256':sha(__file__),
        'scope':'additional conservative reporting check after quality outcomes; no retroactive preregistration',
        'original_coordinate_benchmarks_unchanged':True,'probability_audit_complete_at_registration':(base/'complete.json').exists(),
        'rule':'Require previous frozen-route and conservative-coordinate claim plus full-probability latency/allocated-memory compliance for both policies on every previously frozen common cap. Effective latency upper is max(previous conservative coordinate upper, new probability alpha/16 upper); observed allocated peak is max(old,new). No failed cell may be dropped or rescued.',
        'latency_tail_probability':.05/16,'primary_claim_family':16,
        'new_audit_selection':'Same 13 replacement systems; six direct contextual references are not replacement candidates.',
        'endpoint':'Warm resident decoded adapter through materialized canonical probabilities; setup/load/decode/activation and network serialization excluded.',
        'memory_scope':'Observed peak allocated bytes, not future OOM safety or aggregate coordinator memory.',
        'rich_route':'Already times probability materialization and per-request decode; previous conservative flags pass through unchanged.'})


def run_audit(run):
    rule=run/'protocol/probability_endpoint_report_guard.json'
    assert json.loads(rule.read_text())['analysis_script_sha256']==sha(__file__)
    base=run/'resource_cap_confirmation/probability_endpoint_audit'
    marker=base/'complete.json';done=json.loads(marker.read_text())
    assert done['complete'] and done['post_outcome_additional_guard_not_preregistration']
    assert done['endpoint']=='canonical_four_option_probabilities'
    assert done['planned_blocks']==190 and done['replacement_systems']==13 and done['contextual_direct_references']==6
    for name,h in done['outputs'].items():assert sha(base/name)==h
    assert sha(base/'audit_lock.json')==done['audit_lock_sha256']
    audit_lock=json.loads((base/'audit_lock.json').read_text())
    for name,h in audit_lock['frozen_hashes'].items():assert sha(ROOT/name)==h
    selection_path=run/'resource_cap_confirmation/selection_lock.json'
    assert sha(selection_path)==done['selection_lock_sha256']
    selection=json.loads(selection_path.read_text())
    prior=run/'reports/resource_cap_global_guard'
    old_done=json.loads((prior/'complete.json').read_text());assert old_done['complete']
    for name,h in old_done['inputs_sha256'].items():assert sha(run/name)==h
    for name,h in old_done['outputs_sha256'].items():assert sha(prior/name)==h
    old_systems=pd.read_csv(prior/'system_latency_guard.csv')
    claims=pd.read_csv(prior/'claim_guard.csv')
    probability=pd.read_csv(base/'resource_summary.csv').set_index('system_id')
    assert len(probability)==19 and probability.index.is_unique
    raw_path=base/'measurement/raw_blocks.jsonl'
    raw=[json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    assert len(raw)==190 and len({(r['system_id'],r['timing_block']) for r in raw})==190
    assert set(r['system_id'] for r in raw)==set(probability.index)
    uuids={r.get('runtime',{}).get('GPU_uuid','') for r in raw}
    assert len(uuids)==1 and next(iter(uuids)) not in ('','None',None)
    protocol=json.loads((base/'protocol.json').read_text())
    assert protocol['latency_family_one_sided_divisor']==16
    for sid,summary in probability.iterrows():
        blocks=[r for r in raw if r['system_id']==sid]
        assert len(blocks)==10 and {r['timing_block'] for r in blocks}==set(range(10))
        valid=[r for r in blocks if not r.get('failure_status')]
        assert len(valid)==summary.completed_blocks
        for r in valid:
            assert r['batch_size']==1 and r['context_tokens']==0 and len(r['traces'])==20
            assert r['returned_endpoint']=='canonical_four_option_probabilities'
            assert all(t['probability_finite'] and t['probability_sum_max_error']<1e-12 and t['probability_output_bytes']==32 for t in r['traces'])
            if sid in selection['unique_final_systems']:
                assert r['system_role']=='target_free_replacement' and r['target_called'] is False
        if valid:
            means=np.array([np.mean([t['end_to_end_ms'] for t in r['traces']]) for r in valid])
            rng=np.random.default_rng(protocol['interval_seed'])
            boot=means[rng.integers(len(means),size=(protocol['interval_bootstrap_draws'],len(means)))].mean(1)
            assert abs(float(np.quantile(boot,1-.05/16))-summary.latency_upper_one_sided_alpha_div16)<1e-9
    coordinate_path=run/'resource_cap_confirmation/resource_summary.csv'
    coordinate=pd.read_csv(coordinate_path)
    coordinate=coordinate[(coordinate.batch_size==1)&(coordinate.context_tokens==0)].set_index('system_id')
    rows=[]
    for row in old_systems.to_dict('records'):
        if row['route'].startswith('MCQ'):
            p=probability.loc[row['system_id']]
            complete=p.completed_blocks==10 and p.timed_requests==200
            old_upper=row['global_latency_upper'];new_upper=p.get('latency_upper_one_sided_alpha_div16',np.nan)
            effective=max(old_upper,new_upper) if complete and pd.notna(old_upper) and pd.notna(new_upper) else None
            old_peak=coordinate.loc[row['system_id'],'peak_allocated_bytes'];new_peak=p.get('peak_allocated_bytes',np.nan)
            peak=max(old_peak,new_peak) if complete and pd.notna(old_peak) and pd.notna(new_peak) else None
            row.update(previous_coordinate_latency_upper=old_upper,probability_latency_upper=new_upper,
                global_latency_upper=effective,effective_peak_allocated_bytes=peak,
                full_probability_timing_complete=complete,endpoint='canonical probabilities; warm resident adapter')
        else:
            row.update(full_probability_timing_complete=True,endpoint='full vocabulary probabilities; per-request decode')
        rows.append(row)
    systems=pd.DataFrame(rows);mcq=systems[systems.route.str.startswith('MCQ')].set_index('system_id')
    mapping={(r['target_index'],r['policy'],r['cap_id']):r for r in selection['choices']}
    records=[]
    for row in claims.to_dict('records'):
        earlier=truth(row['global_same_resource_advantage_confirmed'])
        if row['route'].startswith('MCQ'):
            caps=next(p['common_feasible_cap_ids'] for p in selection['power'] if p['target_index']==row['target_index'] and p['comparator']==row['comparator'])
            checks=[]
            for cap in caps:
                for policy in ['budget_aware',row['comparator']]:
                    choice=mapping[row['target_index'],policy,cap];s=mcq.loc[choice['system_id']]
                    checks.append(truth(s.full_probability_timing_complete) and pd.notna(s.global_latency_upper)
                        and s.global_latency_upper<=choice['latency_cap_ms_per_context']
                        and pd.notna(s.effective_peak_allocated_bytes) and s.effective_peak_allocated_bytes<=choice['allocated_VRAM_cap_bytes'])
            resource=bool(checks) and all(checks)
        else:resource=truth(row['global_all_common_resource_caps_confirmed'])
        row.update(previous_conservative_same_resource_advantage_confirmed=earlier,
            full_probability_resource_caps_confirmed=resource,
            full_probability_same_resource_advantage_confirmed=earlier and resource,
            endpoint_audit_scope='Additional post-outcome conservative check; no prior failure rescued')
        records.append(row)
    result=pd.DataFrame(records)
    assert result.primary_registered.map(truth).sum()==16
    assert not (result.full_probability_same_resource_advantage_confirmed & ~result.previous_conservative_same_resource_advantage_confirmed).any()
    out=run/'reports/resource_cap_probability_guard';out.mkdir(parents=True,exist_ok=True)
    systems.to_csv(out/'system_latency_guard.csv',index=False);result.to_csv(out/'claim_guard.csv',index=False)
    inputs=[rule,marker,base/'audit_lock.json',base/'resource_summary.csv',raw_path,base/'protocol.json',selection_path,
        prior/'complete.json',prior/'system_latency_guard.csv',prior/'claim_guard.csv',coordinate_path]
    dump(out/'complete.json',{'complete':True,'additional_post_outcome_check':True,'analysis_script_sha256':sha(__file__),
        'primary_composite_claims':16,'primary_claims_passing':int(result.loc[result.primary_registered.map(truth),'full_probability_same_resource_advantage_confirmed'].sum()),
        'inputs_sha256':{str(p.relative_to(run)):sha(p) for p in inputs},
        'outputs_sha256':{name:sha(out/name) for name in ['system_latency_guard.csv','claim_guard.csv']}})
    print(result.to_string(index=False));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['register','audit']);parser.add_argument('--run-root',type=Path,default=RUN)
    args=parser.parse_args();register(args.run_root) if args.action=='register' else run_audit(args.run_root)
