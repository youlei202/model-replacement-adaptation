#!/usr/bin/env python3
"""Operational monitor and deterministic finalizer for the six frozen settings."""
from pathlib import Path
import json,subprocess,time
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/runs/v2/resource_cap_confirmation'
def main():
 while True:
  done=0;plans=780;queues=[];last=None
  for p in sorted((BASE/'resources').glob('batch*_context*/progress.json')):
   a=json.loads(p.read_text());done+=a['completed_blocks'];last=a.get('last')
  for context in [0,128,512]:
   p=BASE/f'resources/queue_context{context}_status.json';queues.append(json.loads(p.read_text()) if p.exists() else {'stage':'not_started','context_tokens':context})
  state={'stage':'live_resource_measurements','completed_blocks':done,'planned_blocks':plans,'last':last,'quality_complete':True,'updated_at':time.time(),'queues':queues};path=BASE/'run_status.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
  if any(q['stage']=='failed' for q in queues):raise RuntimeError('A whole-setting queue failed; preserve evidence and diagnose before restarting')
  if all(q['stage']=='complete_gpu_released' for q in queues):break
  time.sleep(30)
 raw=[BASE/f'resources/batch{b}_context{c}/raw_blocks.jsonl' for b in [1,8] for c in [0,128,512]]
 commands=[['mcq_resource_cap_analyze.py','resources','--raw',*map(str,raw)],['mcq_resource_cap_contract.py'],['mcq_resource_cap_cost_ledger.py']]
 for i,tail in enumerate(commands):
  cmd=[str(ROOT/'.venv-benchmarks/bin/python'),'-u',str(ROOT/'experiments'/tail[0]),*tail[1:]];start=time.time()
  with (BASE/f'finalize_{i}.log').open('w') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  record={'command':cmd,'start_unix':start,'end_unix':time.time(),'returncode':r.returncode}
  with (BASE/'finalization_commands.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
  print(json.dumps(record),flush=True)
  if r.returncode:raise RuntimeError('Finalization failed; inspect preserved log')
 print('SUPPLEMENT_COMPLETE',flush=True)
if __name__=='__main__':main()
