"""Synthetic failures must remain outcomes and must never pass a cap guard."""
import importlib.util,json,sys
from pathlib import Path
import pandas as pd
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import mcq_probability_endpoint_audit as audit

def test_failed_block_retained_and_caps_fail(tmp_path,monkeypatch):
 cap=tmp_path/'cap';base=cap/'probability_endpoint_audit';measurement=base/'measurement';measurement.mkdir(parents=True)
 monkeypatch.setattr(audit,'BASE',base);monkeypatch.setattr(audit,'CAP',cap);monkeypatch.setattr(audit,'verify_lock',lambda:{'created_unix':0})
 systems=[f'target_free_{i}' for i in range(13)]+[f'direct_{i}' for i in range(6)]
 audit.dump(base/'benchmark_spec.json',{'systems':[{'system_id':s} for s in systems]});audit.dump(base/'audit_lock.json',{});audit.dump(cap/'selection_lock.json',{})
 audit.dump(measurement/'measurement_lock.json',{'spec_sha256':audit.sha(base/'benchmark_spec.json'),'benchmark_code_sha256':audit.sha(audit.WRAPPER),'created_at':'2026-11-10T01:00:00+00:00'})
 blocks=[]
 for s in systems:
  for b in range(10):
   block={'system_id':s,'timing_block':b,'batch_size':1,'context_tokens':0,'runtime':{'GPU_uuid':'synthetic-fixture-uuid'},'system_role':'target_free_replacement' if s.startswith('target') else 'direct_model_contextual_reference'}
   if s==systems[0] and b==2:block['failure_status']='Synthetic failure'
   else:block.update(failure_status=None,returned_endpoint='canonical_four_option_probabilities',warmup_requests=20,timed_requests=20,timing_scope='synthetic fixture',traces=[{'end_to_end_ms':5.,'context_count':1,'truncated_peer_prompts':0,'probability_finite':True,'probability_sum_max_error':0.,'probability_output_bytes':32,'probability_sha256':'0'*64,'peak_allocated_bytes':100,'peak_reserved_bytes':200,'host_RSS':300} for _ in range(20)])
   blocks.append(block)
  audit.dump(measurement/f'{s}.replay.json',{'failure_status':None,'replay_checks':[{'batch_size':1,'max_probability_difference':0.,'max_prediction_coordinate_difference':0.,'probability_finite':True,'probability_sum_max_error':0.}]})
 (measurement/'raw_blocks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in blocks))
 pd.DataFrame([{'target_index':target,'policy':f'p{p}','cap_id':c,'system_id':systems[0],'latency_cap_ms_per_context':10,'allocated_VRAM_cap_bytes':500} for target in (1,3) for p in range(7) for c in range(16)]).to_csv(cap/'cap_quality_results.csv',index=False)
 audit.analyze()
 result=json.loads((base/'complete.json').read_text());assert result['failed_blocks']==1 and result['timed_requests']==3780
 coverage=pd.read_csv(base/'cap_compliance_additional_guard.csv');assert len(coverage)==224 and not coverage.probability_endpoint_point_compliant.any() and not coverage.probability_endpoint_bound_compliant.any()
 failed=json.loads((base/'failed_blocks.json').read_text());assert len(failed)==1 and failed[0]['timing_block']==2
