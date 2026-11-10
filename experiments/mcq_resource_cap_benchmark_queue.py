#!/usr/bin/env python3
"""Operational whole-setting queue; immutable specs/scientific benchmark unchanged."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/runs/v2/resource_cap_confirmation'
def main():
 a=argparse.ArgumentParser();a.add_argument('--gpu',required=True);a.add_argument('--context',type=int,choices=[0,128,512],required=True);args=a.parse_args();output=BASE/'resources';output.mkdir(exist_ok=True);assignment=output/f'queue_context{args.context}_assignment.json'
 assignment.write_text(json.dumps({'created_unix':time.time(),'gpu':args.gpu,'entire_settings':[{'batch_size':b,'context_tokens':args.context} for b in [1,8]],'all13systems_sameGPUwithinsetting':True,'scientific_choices_and_timed_protocol_unchanged':True,'maximum_worker_processes_by_setting':{str(b):json.loads((BASE/('benchmark_execution_specs' if (BASE/'benchmark_execution_specs').exists() else 'benchmark_specs')/f'batch{b}_context{args.context}.json').read_text()).get('maximum_worker_processes',4) for b in [1,8]},'worker_cache_amendment':'execution_worker_cache_amendment.json'},indent=2)+'\n')
 for batch in [1,8]:
  spec=BASE/('benchmark_execution_specs' if (BASE/'benchmark_execution_specs').exists() else 'benchmark_specs')/f'batch{batch}_context{args.context}.json';dest=output/f'batch{batch}_context{args.context}';cmd=[str(ROOT/'.venv-inference/bin/python'),'-u',str(ROOT/'experiments/mcq_resource_benchmark.py'),'--spec',str(spec),'--output',str(dest)];env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,HF_HUB_CACHE='/tmp/model-replacement-adaptation/hf_cache',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false');start=time.time();log=output/f'batch{batch}_context{args.context}.log'
  (output/f'queue_context{args.context}_status.json').write_text(json.dumps({'stage':'running','gpu':args.gpu,'batch_size':batch,'context_tokens':args.context,'start_unix':start})+'\n')
  with log.open('w') as out:result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
  record={'gpu':args.gpu,'batch_size':batch,'context_tokens':args.context,'start_unix':start,'end_unix':time.time(),'returncode':result.returncode,'command':cmd,'spec_sha256':hashlib.sha256(spec.read_bytes()).hexdigest()}
  with (output/f'queue_context{args.context}_commands.jsonl').open('a') as out:out.write(json.dumps(record)+'\n')
  print(json.dumps(record),flush=True)
  if result.returncode:
   (output/f'queue_context{args.context}_status.json').write_text(json.dumps({'stage':'failed','gpu':args.gpu,'batch_size':batch,'context_tokens':args.context,'returncode':result.returncode})+'\n');raise RuntimeError('Benchmark failed, preserve artifacts and diagnose')
 (output/f'queue_context{args.context}_status.json').write_text(json.dumps({'stage':'complete_gpu_released','gpu':args.gpu,'context_tokens':args.context,'end_unix':time.time()})+'\n');print('GPU_RELEASED',args.gpu,flush=True)
if __name__=='__main__':main()
