#!/usr/bin/env python3
"""Only tighten the frozen cap-route claims with a common family-16 latency guard.

The rule was prescribed by the coordinator before the rich quality marker; its
implementation hash was recorded later. It is labeled an additional conservative
reporting check, not a fully pretest code freeze. It never changes candidates,
methods, cap sets, query cohorts, or original route results.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments/runs/v2'
FIXED = ('fixed_native_full', 'fixed_unbounded_best', 'fixed_mean_budget',
         'fixed_minimax_validation_regret')
SEED = 81724099
COMPARISONS = 16
DRAWS = 10000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def upper_bound(block_means, frozen_route_upper):
    """The maximum makes this audit monotone even with bootstrap MC variation."""
    means=np.asarray(block_means,dtype=float)
    if len(means)!=10 or not np.isfinite(means).all() or (means<=0).any():
        return None
    rng=np.random.default_rng(SEED)
    boot=means[rng.integers(len(means),size=(DRAWS,len(means)))].mean(1)
    strict=float(np.quantile(boot,1-.05/COMPARISONS))
    return max(float(frozen_route_upper),strict)


def register(run,allow_late=False):
    path=run/'protocol/global_resource_cap_report_guard.json'
    if path.exists():
        assert json.loads(path.read_text())['analysis_script_sha256']==sha(__file__)
        print('Existing registration unchanged');return
    # Quality outcome evaluation, not mere response collection, is the boundary.
    forbidden=[run/'resource_cap_confirmation/quality_complete.json',
               run/'rich_behavior/matched_resource_supplement/quality_complete.json']
    existing=[p for p in forbidden if p.exists()]
    if existing and not allow_late:raise RuntimeError('Cannot claim a pretest registration after supplemental outcome analysis; explicitly label an additional conservative check')
    dump(path,{'registered_unix':time.time(),'analysis_script_sha256':sha(__file__),
        'primary_composite_claims':COMPARISONS,'alpha':.05,'latency_tail_probability':.05/COMPARISONS,
        'bootstrap_draws':DRAWS,'bootstrap_seed':SEED,
        'registration_scope':'additional conservative reporting check' if existing else 'before supplemental quality markers',
        'fully_pretest_code_freeze':not existing,
        'chronology':'Coordinator prescribed the .05/16 rule in the conversation before the rich quality marker. Executable hash recorded after that marker. No new quality values were consulted to choose this rule; original frozen analyses remain unchanged.' if existing else 'Registered before supplemental quality markers exist.',
        'quality_markers_already_present':[{ 'path':str(p.relative_to(run)), 'sha256':sha(p), 'mtime_unix':p.stat().st_mtime } for p in existing],
        'rule':'For every selected natural batch1 system, resample its ten timing-block mean complete-predictor wall latencies. Use one-sided quantile 1-.05/16. Final bound is max(this bound, frozen route-specific bound), so no prior failure can become a pass.',
        'claim':'Preserve original quality/argmax criteria and require both systems to meet every pre-frozen common latency/allocated-VRAM cap. Primary global claim = original frozen route claim AND stricter resource guard.',
        'original_outputs':'Read only; original route bounds and flags remain visible.',
        'memory_scope':'Observed peak allocated VRAM on the measured fresh workload only, not a guarantee about future OOM or reserved memory.',
        'uncertainty_scope':'Approximate block bootstrap for mean latency. Extra fixed/cheap contrasts remain descriptive; no extra primary claims are added.'})
    print(path)


def run_audit(run):
    rule=run/'protocol/global_resource_cap_report_guard.json'
    assert json.loads(rule.read_text())['analysis_script_sha256']==sha(__file__)
    out=run/'reports/resource_cap_global_guard';out.mkdir(parents=True,exist_ok=True)
    inputs={str(rule.relative_to(run)):sha(rule)};system_rows=[];claim_rows=[]
    routes=[('MCQ measured resource caps',run/'resource_cap_confirmation',False),
            ('Rich measured resource caps',run/'rich_behavior/matched_resource_supplement',True)]
    for route,base,rich in routes:
        marker=base/('supplement_complete.json' if rich else 'complete.json')
        complete=json.loads(marker.read_text());assert complete['complete'] is True
        for field in ('outputs','outputs_sha256','quality_outputs'):
            for name,expected in complete.get(field,{}).items():assert sha(base/name)==expected
        lockpath=base/('prospective_lock.json' if rich else 'selection_lock.json')
        lock=json.loads(lockpath.read_text())
        rawpaths=[base/'timing_raw.jsonl'] if rich else [Path(p) for p in complete['raw_measurements']]
        for path in [marker,lockpath,*rawpaths]:inputs[str(path.relative_to(run))]=sha(path)
        frames=[]
        if rich:
            raw=pd.read_json(rawpaths[0],lines=True)
            for sid,g in raw[(raw.batch_size==1)&(raw.context_tokens==0)].groupby('system_digest'):
                if g.duplicated(['timing_block','request']).any():raise AssertionError('Duplicated rich timing request')
                counts=g.groupby('timing_block').size()
                means=g.groupby('timing_block').end_to_end_ms.mean().sort_index().to_numpy() if len(counts)==10 and counts.eq(20).all() else []
                frames.append((sid,means))
        else:
            groups={}
            for path in rawpaths:
                assert sha(path)==complete['raw_measurements'][str(path)]
                for line in path.read_text().splitlines():
                    b=json.loads(line)
                    if b['batch_size']!=1 or b['context_tokens']!=0:continue
                    groups.setdefault(b['system_id'],[]).append(b)
            for sid,rows in groups.items():
                if len({b['timing_block'] for b in rows})!=len(rows):raise AssertionError('Duplicated MCQ timing block')
                good=sorted([b for b in rows if not b.get('failure_status')],key=lambda b:b['timing_block'])
                means=[np.mean([t['end_to_end_ms']/t['context_count'] for t in b['traces']]) for b in good] if len(good)==10 and all(len(b['traces'])==20 for b in good) else []
                frames.append((sid,means))
        summarypath=base/'resource_summary.csv';summary=pd.read_csv(summarypath)
        nat=summary[(summary.batch_size==1)&(summary.context_tokens==0)].set_index('candidate_id' if rich else 'system_id')
        bounds={}
        for sid,means in frames:
            old=nat.loc[sid,'mean_wall_upper_simultaneous' if rich else 'latency_familywise_upper_one_sided']
            bound=upper_bound(means,old) if pd.notna(old) else None
            bounds[sid]=bound
            system_rows.append({'route':route,'system_id':sid,'successful_blocks':len(means),
                'frozen_route_latency_upper':float(old) if pd.notna(old) else None,
                'global_latency_upper':bound,'one_sided_tail_probability':.05/COMPARISONS,
                'global_guard_can_only_tighten':bound is None or bound>=old})
        source=base/('primary_confirmation.csv' if rich else 'guarded_confirmation_results.csv')
        result=pd.read_csv(source);inputs[str(source.relative_to(run))]=sha(source)
        inputs[str(summarypath.relative_to(run))]=sha(summarypath)
        decisions=lock['decisions'] if rich else lock['choices']
        mapping={(r['target_index'],r['policy'],r['cap_id']):r for r in decisions}
        for row in result.to_dict('records'):
            target=row['target_index'];policy=row['fixed_policy' if rich else 'comparator']
            caps=row['common_feasible_caps'].split('|') if rich and isinstance(row.get('common_feasible_caps'),str) else []
            if not rich:caps=next(x['common_feasible_cap_ids'] for x in lock['power'] if x['target_index']==target and x['comparator']==policy)
            checks=[]
            for cap in caps:
                for name in ['joint' if rich else 'budget_aware',policy]:
                    point=mapping[target,name,cap];sid=point['candidate_id' if rich else 'system_id'];bound=bounds.get(sid)
                    latency_cap=point['mean_wall_latency_cap_ms' if rich else 'latency_cap_ms_per_context']
                    observed_peak=nat.loc[sid,'peak_allocated_bytes']
                    checks.append(bound is not None and bound<=latency_cap and observed_peak<=point['allocated_VRAM_cap_bytes'])
            old=str(row.get('same_resource_advantage_confirmed')).lower() in ('true','1','1.0');guard=bool(checks) and all(checks)
            claim_rows.append({'route':route,'target_index':target,'comparator':policy,
                'primary_registered':policy in FIXED,'common_caps':len(caps),
                'frozen_route_same_resource_advantage_confirmed':old,
                'global_all_common_resource_caps_confirmed':guard,
                'global_same_resource_advantage_confirmed':old and guard,
                'memory_scope':'observed peak allocated VRAM only; no future OOM guarantee',
                'source_file':str(source.relative_to(run))})
    systems=pd.DataFrame(system_rows);claims=pd.DataFrame(claim_rows)
    assert int(claims.primary_registered.sum())==16
    systems.to_csv(out/'system_latency_guard.csv',index=False)
    claims.to_csv(out/'claim_guard.csv',index=False)
    dump(out/'complete.json',{'complete':True,'analysis_script_sha256':sha(__file__),
         'registered_rule_sha256':sha(rule),'primary_composite_claims':16,
         'primary_claims_passing':int(claims.loc[claims.primary_registered,'global_same_resource_advantage_confirmed'].sum()),
         'inputs_sha256':inputs,'outputs_sha256':{name:sha(out/name) for name in ['system_latency_guard.csv','claim_guard.csv']}})
    print(claims.to_string(index=False));return claims


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['register','audit']);parser.add_argument('--run-root',type=Path,default=RUN);parser.add_argument('--additional-conservative-check',action='store_true')
    args=parser.parse_args();register(args.run_root,args.additional_conservative_check) if args.action=='register' else run_audit(args.run_root)
