#!/usr/bin/env python3
"""Durable operational wait/claim/run/finalize for separately locked audit."""
from pathlib import Path
import json,os,subprocess,time
from mcq_probability_endpoint_audit import BASE,CAP,ROOT,sha,verify_lock,dump

def memory():
 root=Path('/sys/fs/cgroup');stat={k:int(v) for k,v in (x.split() for x in (root/'memory.stat').read_text().splitlines())}
 return {'observed_unix':time.time(),'memory_current':int((root/'memory.current').read_text()),'memory_max':int((root/'memory.max').read_text()),'anon':stat['anon'],'file':stat['file']}
def main():
 verify_lock()
 if (BASE/'execution_assignment.json').exists():raise RuntimeError('Assignment already exists; preserve prior run')
 while True:
  available=[]
  for context in (0,128,512):
   path=CAP/f'resources/queue_context{context}_status.json'
   if path.exists():
    row=json.loads(path.read_text())
    if row.get('stage')=='complete_gpu_released':available.append(row)
  for row in sorted(available,key=lambda x:x['end_unix']):
   gpu=str(row['gpu']);usage=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();used,util=map(int,usage.split(','))
   if used>64 or util>0:continue
   snapshot=memory();budget=json.loads((BASE/'memory_plan.json').read_text())['audit_full_cache_weights_bytes']
   if snapshot['anon']+budget*1.25>snapshot['memory_max']*.9:continue
   assignment={'created_unix':time.time(),'gpu':gpu,'host':__import__('socket').gethostname(),'previous_queue_release':row,'verified_gpu_memory_MiB':used,'verified_gpu_utilization_percent':util,'host_memory':snapshot,'planned_full_cache_weight_bytes':budget,'audit_spec_sha256':sha(BASE/'benchmark_spec.json'),'audit_lock_sha256':sha(BASE/'audit_lock.json'),'all19systems_sameGPU':True,'cold_setup_activation_decode_excluded_from_warm_endpoint':True}
   dump(BASE/'execution_assignment.json',assignment);break
  else:
   dump(BASE/'run_status.json',{'stage':'waiting_for_whole_coordinate_queue_release','updated_at':time.time(),'available_completed_queues':available});print('WAITING_FOR_RELEASE',flush=True);time.sleep(30);continue
  break
 verify_lock();cmd=[str(ROOT/'.venv-inference/bin/python'),'-u',str(ROOT/'experiments/mcq_probability_endpoint_benchmark.py'),'--spec',str(BASE/'benchmark_spec.json'),'--output',str(BASE/'measurement')]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,HF_HUB_CACHE='/tmp/model-replacement-adaptation/hf_cache',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false');start=time.time();dump(BASE/'run_status.json',{'stage':'running_probability_endpoint_audit','gpu':gpu,'start_unix':start,'planned_blocks':190});print('AUDIT_GPU_CLAIMED',gpu,flush=True)
 with (BASE/'benchmark.log').open('w') as out:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
 record={'command':cmd,'start_unix':start,'end_unix':time.time(),'returncode':r.returncode,'GPU':gpu,'spec_sha256':sha(BASE/'benchmark_spec.json')};dump(BASE/'execution_command.json',record)
 if r.returncode:
  dump(BASE/'run_status.json',{'stage':'failed','error':'Benchmark exited '+str(r.returncode),'GPU':gpu});raise RuntimeError('Probability endpoint audit failed; preserve all artifacts')
 dump(BASE/'gpu_release.json',{'GPU':gpu,'end_unix':time.time(),'benchmark_workers_exited':True});print('AUDIT_GPU_RELEASED',gpu,flush=True)
 cmd=[str(ROOT/'.venv-benchmarks/bin/python'),str(ROOT/'experiments/mcq_probability_endpoint_audit.py'),'analyze'];start=time.time()
 with (BASE/'analysis.log').open('w') as out:r=subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
 dump(BASE/'analysis_command.json',{'command':cmd,'start_unix':start,'end_unix':time.time(),'returncode':r.returncode})
 if r.returncode:
  dump(BASE/'run_status.json',{'stage':'analysis_failed','error':'Analyzer exited '+str(r.returncode)});raise RuntimeError('Probability endpoint analysis failed')
 dump(BASE/'run_status.json',{'stage':'complete','updated_at':time.time(),'complete':True,'completed_blocks':190,'planned_blocks':190});print('AUDIT_COMPLETE',flush=True)
if __name__=='__main__':main()
