#!/usr/bin/env python3
"""Operational watcher only; launch the already-frozen analysis once all six banks complete."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/runs/v2/resource_cap_confirmation'
def main():
 while True:
  status=json.loads((BASE/'run_status.json').read_text())
  if status.get('stage')=='collection_failed':raise RuntimeError('Collection failed; analysis not run')
  files=list((BASE/'response_cache_test').glob('*.provenance.json'))
  if len(files)==6 and all(json.loads(p.read_text()).get('completed') for p in files) and status.get('new_test_collection_complete'):break
  time.sleep(20)
 command=[str(ROOT/'.venv-benchmarks/bin/python'),'-u',str(ROOT/'experiments/mcq_resource_cap_analyze.py'),'quality'];started=time.time()
 with (BASE/'quality_analysis.log').open('w') as out:result=subprocess.run(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
 (BASE/'quality_analysis_command.json').write_text(json.dumps({'command':command,'started_unix':started,'ended_unix':time.time(),'returncode':result.returncode},indent=2)+'\n')
 if result.returncode:raise RuntimeError('Frozen analysis failed; diagnostic retained')
 print('QUALITY_COMPLETE',flush=True)
if __name__=='__main__':main()
