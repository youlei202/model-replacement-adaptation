#!/usr/bin/env python3
"""Operational case partition: pause only our queue parent, not its scorer child.

The immutable original queue later skips completed cases using the unchanged
collector's completed-bank identity check. No same-model output is raced.
"""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/runs/v2/resource_cap_confirmation'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--parent-pid',type=int,required=True);ap.add_argument('--gpu',required=True);a=ap.parse_args();commandline=Path(f'/proc/{a.parent_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'mcq_resource_cap_collect.py' in commandline and '--gpu 0' in commandline
 record={'created_unix':time.time(),'parent_pid':a.parent_pid,'parent_command':commandline,'GPU0_model_cases':[3,1,0],'GPU2_model_cases':[2,4,5],'existing_parent_later_skips_complete_GPU2_cases':True,'pause_only_queue_parent_scorer_child_continues':True,'scientific_code_and_locks_unchanged':True,'uniform_runtime':True};(BASE/'parallel_collection_case_assignment.json').write_text(json.dumps(record,indent=2)+'\n');os.kill(a.parent_pid,signal.SIGSTOP)
 try:
  for index in [2,4,5]:
   cmd=[str(ROOT/'.venv-inference/bin/python'),'-u',str(ROOT/'experiments/confirmation_collect.py'),'--model-index',str(index),'--stage','test','--data-dir',str(BASE)];env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,HF_HUB_CACHE='/tmp/model-replacement-adaptation/hf_cache',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false');start=time.time()
   (BASE/'parallel_collection_status.json').write_text(json.dumps({'stage':'collecting','active_model':index,'gpu':a.gpu,'updated_unix':start})+'\n')
   with (BASE/'logs'/f'model_{index}_gpu2.log').open('a') as log:run=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   outcome={'model_global_id':index,'gpu':a.gpu,'start_unix':start,'end_unix':time.time(),'returncode':run.returncode,'command':cmd}
   with (BASE/'parallel_collection_commands.jsonl').open('a') as out:out.write(json.dumps(outcome)+'\n')
   print(json.dumps(outcome),flush=True)
   if run.returncode:raise RuntimeError('Parallel case failed; preserved diagnostic and main queue resumes')
  (BASE/'parallel_collection_status.json').write_text(json.dumps({'stage':'complete_gpu_released','gpu':a.gpu,'completed_models':[2,4,5],'updated_unix':time.time()})+'\n');print('GPU2_RELEASED',flush=True)
 finally:os.kill(a.parent_pid,signal.SIGCONT)
if __name__=='__main__':main()
