"""Synthetic guard regression: resource failures never change the query endpoint."""
import json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import mcq_resource_cap_analyze as module

def test_failed_cap_preserves_registered_quality_endpoint(tmp_path,monkeypatch):
 choices=[]
 for cap in ['tight','loose','unavailable']:
  for policy,system in [('budget_aware','adaptive'),('fixed_mean_budget','fixed')]:
   choices.append({'target_index':3,'policy':policy,'cap_id':cap,
                   'system_id':system if cap!='unavailable' else None,
                   'status':'eligible' if cap!='unavailable' else 'infeasible',
                   'latency_cap_ms_per_context':70.,'allocated_VRAM_cap_bytes':4096})
 locked={'choices':choices,'unique_final_systems':['adaptive','fixed'],'underpowered':False,
         'power':[{'target_index':3,'comparator':'fixed_mean_budget',
                   'common_feasible_cap_ids':['tight','loose'],
                   'common_feasible_cap_count':2,'primary':True}]}
 monkeypatch.setattr(module,'BASE',tmp_path);monkeypatch.setattr(module,'verify_lock',lambda:locked)
 (tmp_path/'selection_lock.json').write_text(json.dumps(locked));(tmp_path/'quality_complete.json').write_text(json.dumps({'outputs':{},'fresh_clusters':2768}))
 rows=[{k:v for k,v in r.items() if k!='candidate'} for r in locked['choices']];pd.DataFrame(rows).to_csv(tmp_path/'cap_quality_results.csv',index=False)
 contrasts=[]
 for power in locked['power']:
  contrasts.append({'target_index':power['target_index'],'comparator':power['comparator'],'number_common_caps':power['common_feasible_cap_count'],'gain_tv':.01,'TV_and_argmax_statistical_criterion_met':True,'statistical_effect_criterion_met':True,'primary':power['primary']})
 pd.DataFrame(contrasts).to_csv(tmp_path/'all_comparator_results.csv',index=False)
 for batch in [1,8]:
  for context in [0,128,512]:
   path=tmp_path/f'b{batch}_c{context}.jsonl'
   with path.open('w') as out:
    for system in locked['unique_final_systems']:
     for block in range(10):
      # Every natural case violates latency. Quality numbers must survive unchanged.
      ms=1e6 if batch==1 and context==0 else 10
      record={'system_id':system,'batch_size':batch,'context_tokens':context,'timing_block':block,'failure_status':None,'warmup_requests':20,'model_parameter_and_buffer_bytes':100,'actual_call_q':2,'adapter_accounting':{},'traces':[{'end_to_end_ms':ms,'context_count':batch,'peak_allocated_bytes':1024,'peak_reserved_bytes':2048,'host_RSS':4096} for _ in range(20)]};out.write(json.dumps(record)+'\n')
 module.resources(sorted(tmp_path.glob('*.jsonl')))
 result=pd.read_csv(tmp_path/'guarded_confirmation_results.csv');assert len(result)==len(contrasts)
 assert result.gain_tv.eq(.01).all()
 assert result.number_common_caps.tolist()==[r['number_common_caps'] for r in contrasts]
 assert not result.same_resource_advantage_confirmed.any()
 assert not result.all_common_caps_fresh_confirmed.any()
 coverage=pd.read_csv(tmp_path/'coverage_and_cap_compliance.csv');assert len(coverage)==len(rows)
 assert coverage.status.eq('infeasible').sum()==sum(r['status']=='infeasible' for r in rows)
 assert not coverage.final_measured_point_cap_compliant.any()
