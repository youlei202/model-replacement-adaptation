"""Validate complete new banks and execute all q=2 fresh-cohort candidate fits."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--stage',choices=['fit','evaluate'],default='fit');args=a.parse_args();base=ROOT/'experiments/runs/v2/confirmation';lock=json.loads((base/'data_lock.json').read_text());reg=json.loads((ROOT/'experiments/experiment_config/configs/campaign.json').read_text())['model_registry'];role='fitval' if args.stage=='fit' else 'test';required=pd.read_parquet(base/f'required_items_{role}.parquet');expected=set(map(tuple,required[['task_id','item_id','prompt_hash']].to_numpy()));runtime=None;records=[]
 for r in reg:
  p=base/f'response_cache_{role}'/(r['model_id'].replace('/','--')+'.parquet');prov=json.loads(p.with_suffix('.provenance.json').read_text());d=pd.read_parquet(p)
  assert prov['completed'] and prov['official_forward_check']['passed'];assert prov['revision']==r['revision'];assert prov['response_bank_sha256']==sha(p);assert set(d.model_id)=={r['model_id']} and set(d.revision)=={r['revision']};assert set(map(tuple,d[['task_id','item_id','prompt_hash']].to_numpy()))==expected;assert len(d)==len(required)
  if runtime is None:runtime=prov['runtime']
  else:assert runtime==prov['runtime']
  records.append({'model_id':r['model_id'],'rows':len(d),'revision':r['revision'],'bank_sha256':sha(p),'provenance_sha256':sha(p.with_suffix('.provenance.json')),'historical_bridge_passed':(prov.get('historical_runtime_bridge') or {}).get('passed')})
 audit={'role':role,'required_unique_queries':len(required),'total_responses':len(required)*6,'same_runtime_all_six':True,'runtime':runtime,'models':records};(base/f'{role}_cache_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 targetids=sorted(set(s['target_global_id'] for s in lock['settings']));tasks=sorted(set(s['task_id'] for s in lock['settings']))
 if len(tasks)>1:raise RuntimeError('Explicit per-setting cross-product guard: current shared runner supports both targets on one external population; extend with registered cells if two tasks are selected')
 command=[str(ROOT/'.venv-benchmarks/bin/python'),'-u',str(ROOT/'experiments/phase_a_run.py'),'--manifest-dir',str(base/'manifests'),'--response-dir',str(base/'response_cache_fitval'),'--test-response-dir',str(base/'response_cache_test'),'--dimensions-json',str(base/'dimensions.json'),'--targets',*[str(t) for t in targetids],'--seeds',*[str(s) for s in lock['training_seeds']],'--tasks',*tasks,'--q','2','--stage',args.stage,'--workers','6','--blas-threads','4','--bootstrap-draws','10000','--output',str(base/'fits/campaign'),'--data-role','external_confirmation_fresh_four_choice_population']
 start=time.time();env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4');path=base/f'{args.stage}_runner.log'
 with path.open('a') as f:p=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 record={'stage':args.stage,'command':command,'start_unix':start,'end_unix':time.time(),'returncode':p.returncode,'log':str(path.relative_to(base))}
 with (base/'fit_evaluation_commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 print(json.dumps(record,indent=2));raise SystemExit(p.returncode)
if __name__=='__main__':main()
