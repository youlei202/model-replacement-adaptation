"""Durable, resumable rich-behavior execution on the assigned B200, no silent skipped stages."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from rich_setup import OUT,MODELS,digest,dump

def status(stage,**kw):dump(OUT/'run_status.json',{'stage':stage,'time_unix':time.time(),**kw})
def run(script,*args):
 cmd=[sys.executable,str(ROOT/'experiments'/script),*args]
 print('RUN',json.dumps(cmd),flush=True);subprocess.run(cmd,cwd=ROOT,check=True)

def pilot_report():
 from scipy.special import softmax,log_softmax
 metas=[json.load(open(OUT/'logits/pilot'/f'model_{i}.json')) for i in range(5)]
 arrays=[np.load(OUT/'logits/pilot'/f'model_{i}.npy',mmap_mode='r') for i in range(5)]
 assert all(a.shape==(128,151936) for a in arrays)
 rows=[]
 for target in [3,4]:
  for peer in [0,1,2]:
   tv=[]
   for s in range(0,128,8):
    p=softmax(np.asarray(arrays[target][s:s+8],np.float64),axis=1);q=softmax(np.asarray(arrays[peer][s:s+8],np.float64),axis=1);tv.extend((.5*np.abs(p-q).sum(1)).tolist())
   rows.append({'target_index':target,'peer_index':peer,'mean_tv':float(np.mean(tv)),'sd_tv':float(np.std(tv,ddof=1))})
 # Dense small-position FP64 reference check against chunked full-vocabulary implementation.
 sys.path.insert(0,str(ROOT/'experiments/experiment_config'))
 from reference.full_vocab import exact_metrics
 a,b=np.asarray(arrays[3][:3],np.float64),np.asarray(arrays[0][:3],np.float64);direct=.5*np.abs(softmax(a,axis=1)-softmax(b,axis=1)).sum(1);chunk=exact_metrics(a,b,batch_positions=1)['tv']
 assert np.max(np.abs(direct-chunk))<1e-12
 report={'pilot_complete':True,'models':5,'documents_per_model':128,'model_context_evaluations':640,'official_extra_forward_calls':10,'pilot_safety_cap':768,'exact_dense_metric_max_difference':float(np.max(np.abs(direct-chunk))),'common_output_width':151936,'all_official_last_logits_checks_exact':all(max(m['official_last_position_max_difference'])==0 for m in metas),'storage':'float32 centered logits','fp16_storage_error_by_model':{m['model_id']:m['fp16_storage_error_tv'] for m in metas},'direct_peer_diagnostics':rows,'largest_model_peak_vram_bytes':max(m['memory_peak_allocated'] for m in metas),'pilot_total_load_and_inference_seconds':sum(m['seconds_including_load'] for m in metas),'projected_full_development_logit_bytes':5*3072*151936*4,'projected_full_development_seconds_linear_context_estimate':sum(sum(x['seconds'] for x in m['batch_timings']) for m in metas)*24,'numerical_error_not_used_to_dropcoordinates':True,'scope':'development diagnostics only; no independentconfirmation'}
 dump(OUT/'rich_pilot_report.json',report);print(json.dumps(report),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--through',choices=['pilot','fit','evaluation','resource'],default='evaluation');a=p.parse_args()
 try:
  status('waiting_for_data_lock')
  while not (OUT/'data_lock.json').exists():time.sleep(10)
  for role in ['pilot','fit','validation']:
   status('collecting_'+role)
   for i in range(5):run('rich_collect.py','--model-index',str(i),'--role',role,'--device','cuda:0')
   if role=='pilot':
    pilot_report()
    if a.through=='pilot':status('pilot_complete');return
  status('fitting_full_candidate_family');run('rich_fit.py','fit')
  status('profiling_live_development_systems');run('rich_deploy.py','profile')
  status('freezing_selection');run('rich_evaluate.py','freeze')
  status('freezing_resource_operating_points');run('rich_deploy.py','freeze')
  if a.through=='fit':status('fit_and_selection_frozen');return
  status('collecting_sealed_test')
  for i in range(5):run('rich_collect.py','--model-index',str(i),'--role','test','--device','cuda:0')
  status('evaluating_once');run('rich_evaluate.py','evaluate')
  if a.through=='resource':
   status('live_resource_measurement');run('rich_deploy.py','benchmark')
   status('direct_resource_references');run('rich_deploy.py','direct')
   status('reporting');run('rich_report.py')
  status('complete',through=a.through,scientific_outcome='inspect_measured_results_not_implied_by_exit_status')
 except Exception as e:
  status('failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
