"""A later probability endpoint must never rescue an earlier failed claim."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import probability_endpoint_report_guard as guard


def test_probability_guard_only_tightens_and_rich_passes_through(tmp_path):
    def dump(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    base=tmp_path/'resource_cap_confirmation/probability_endpoint_audit';base.mkdir(parents=True)
    ids=[f's{i}' for i in range(13)]+[f'direct{i}' for i in range(6)]
    policies=['fixed_native_full','fixed_unbounded_best','fixed_mean_budget','fixed_minimax_validation_regret']
    choices=[];power=[];claims=[]
    for target in [1,3]:
        for policy in ['budget_aware',*policies]:
            choices.append({'target_index':target,'policy':policy,'cap_id':'C','system_id':'s0' if policy=='budget_aware' else 's1',
                'latency_cap_ms_per_context':6.,'allocated_VRAM_cap_bytes':100.})
        for policy in policies:
            power.append({'target_index':target,'comparator':policy,'common_feasible_cap_ids':['C']})
            for route in ['MCQ measured resource caps','Rich measured resource caps']:
                earlier=(policy!='fixed_native_full') if route.startswith('MCQ') else True
                claims.append({'route':route,'target_index':target,'comparator':policy,'primary_registered':True,
                    'global_same_resource_advantage_confirmed':earlier,'global_all_common_resource_caps_confirmed':True})
    selection=base.parent/'selection_lock.json';dump(selection,{'choices':choices,'power':power,'unique_final_systems':ids[:13]})
    dump(base/'audit_lock.json',{'frozen_hashes':{}})
    dump(base/'protocol.json',{'latency_family_one_sided_divisor':16,'interval_seed':81724053,'interval_bootstrap_draws':10000})
    raw=[];summaries=[]
    for sid in ids:
        latency=7. if sid=='s0' else 4.
        role='target_free_replacement' if sid.startswith('s') else 'direct_model_contextual_reference'
        for block in range(10):
            raw.append({'system_id':sid,'timing_block':block,'failure_status':None,'batch_size':1,'context_tokens':0,
                'runtime':{'GPU_uuid':'synthetic_gpu'},'system_role':role,'target_called':False,
                'returned_endpoint':'canonical_four_option_probabilities',
                'traces':[{'end_to_end_ms':latency,'probability_finite':True,'probability_sum_max_error':0.,'probability_output_bytes':32} for _ in range(20)]})
        summaries.append({'system_id':sid,'completed_blocks':10,'timed_requests':200,'latency_upper_one_sided_alpha_div16':latency,'peak_allocated_bytes':10.})
    rawpath=base/'measurement/raw_blocks.jsonl';rawpath.parent.mkdir();rawpath.write_text(''.join(json.dumps(r)+'\n' for r in raw))
    pd.DataFrame(summaries).to_csv(base/'resource_summary.csv',index=False)
    dump(base/'complete.json',{'complete':True,'post_outcome_additional_guard_not_preregistration':True,
        'endpoint':'canonical_four_option_probabilities','planned_blocks':190,'replacement_systems':13,'contextual_direct_references':6,
        'audit_lock_sha256':guard.sha(base/'audit_lock.json'),'selection_lock_sha256':guard.sha(selection),
        'outputs':{name:guard.sha(base/name) for name in ['measurement/raw_blocks.jsonl','resource_summary.csv']}})
    prior=tmp_path/'reports/resource_cap_global_guard';prior.mkdir(parents=True)
    pd.DataFrame(claims).to_csv(prior/'claim_guard.csv',index=False)
    systems=[{'route':'MCQ measured resource caps','system_id':sid,'global_latency_upper':5.} for sid in ids[:13]]
    systems.append({'route':'Rich measured resource caps','system_id':'rich','global_latency_upper':3.})
    pd.DataFrame(systems).to_csv(prior/'system_latency_guard.csv',index=False)
    dump(prior/'complete.json',{'complete':True,'inputs_sha256':{},'outputs_sha256':{name:guard.sha(prior/name) for name in ['claim_guard.csv','system_latency_guard.csv']}})
    pd.DataFrame([{'system_id':sid,'batch_size':1,'context_tokens':0,'peak_allocated_bytes':10.} for sid in ids[:13]]).to_csv(base.parent/'resource_summary.csv',index=False)
    guard.register(tmp_path)
    result=guard.run_audit(tmp_path)
    assert not result[result.route.str.startswith('MCQ')].full_probability_same_resource_advantage_confirmed.any()
    assert result[result.route.str.startswith('Rich')].full_probability_same_resource_advantage_confirmed.all()
    new=pd.read_csv(tmp_path/'reports/resource_cap_probability_guard/system_latency_guard.csv').set_index('system_id')
    assert new.loc['s0','global_latency_upper']==7.
    assert new.loc['s1','global_latency_upper']==5.  # A faster later run cannot lower the earlier bound.
