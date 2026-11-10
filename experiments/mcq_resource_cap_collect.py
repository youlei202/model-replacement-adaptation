#!/usr/bin/env python3
"""Sequential six-model inference for the separately locked second MCQ cohort."""
import argparse,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/runs/v2/resource_cap_confirmation'
def status(value):
 path=BASE/'run_status.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value|{'updated_at':time.time()},indent=2)+'\n');tmp.replace(path)
def main():
 a=argparse.ArgumentParser();a.add_argument('--gpu',required=True);args=a.parse_args();logs=BASE/'logs';logs.mkdir(exist_ok=True)
 completed=[]
 for index in [3,1,0,2,4,5]:
  status({'stage':'collecting_second_untouched_cohort','active_model':index,'completed_models':completed,'gpu':args.gpu})
  cmd=[str(ROOT/'.venv-inference/bin/python'),'-u',str(ROOT/'experiments/confirmation_collect.py'),'--model-index',str(index),'--stage','test','--data-dir',str(BASE)]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,HF_HUB_CACHE='/tmp/model-replacement-adaptation/hf_cache',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false');start=time.time()
  with (logs/f'model_{index}.log').open('a') as f:run=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  record={'model_global_id':index,'gpu':args.gpu,'start_unix':start,'end_unix':time.time(),'returncode':run.returncode,'command':cmd}
  with (BASE/'collection_commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  print(json.dumps(record),flush=True)
  if run.returncode:
   status({'stage':'collection_failed','active_model':index,'completed_models':completed,'error':'Model subprocess failed; see log'});raise RuntimeError('Model collection failed; preserved diagnostic log')
  completed.append(index)
 status({'stage':'collection_complete_gpu_released','completed_models':completed,'gpu':args.gpu,'new_test_collection_complete':True});print('ALL_SIX_MODELS_COMPLETE_GPU_RELEASED',flush=True)
if __name__=='__main__':main()
