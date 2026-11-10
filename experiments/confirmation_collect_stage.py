"""Bounded two-GPU response collection; one pinned model per GPU at a time."""
from pathlib import Path
import argparse,concurrent.futures,json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
def main():
 a=argparse.ArgumentParser();a.add_argument('--stage',choices=['fitval','test'],required=True);a.add_argument('--gpus',nargs='+',default=['1','2']);args=a.parse_args();base=ROOT/'experiments/runs/v2/confirmation';folder=base/'logs';folder.mkdir(exist_ok=True)
 jobs=[[3,0],[1,2,4,5]] if len(args.gpus)==2 else [list(range(6))]
 if len(args.gpus)!=len(jobs):raise ValueError('Use1or2GPUs')
 def run(sequence):
  gpu,models=sequence
  for index in models:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_CACHE='/tmp/model-replacement-adaptation/hf_cache',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false');log=folder/f'{args.stage}_model_{index}.log';command=[str(ROOT/'.venv-inference/bin/python'),'-u',str(ROOT/'experiments/confirmation_collect.py'),'--model-index',str(index),'--stage',args.stage];start=time.time()
   with log.open('a') as f:
    f.write(f'\nSTART {start} GPU{gpu}\n');f.flush();p=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
   result={'stage':args.stage,'model_global_id':index,'gpu':gpu,'start_unix':start,'end_unix':time.time(),'returncode':p.returncode,'log':str(log.relative_to(base)),'command':command}
   with (base/'collection_commands.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
   print(json.dumps(result),flush=True)
   if p.returncode:raise RuntimeError(f'Model{index}failed; retainedlog{log}')
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:list(ex.map(run,zip(args.gpus,jobs)))
if __name__=='__main__':main()
