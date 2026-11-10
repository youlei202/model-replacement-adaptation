"""Seal all fresh fitted methods, analysis and operational choices before test."""
from pathlib import Path
import argparse,hashlib,json,os,time,shutil
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=ROOT/'experiments/runs/v2/confirmation');a.add_argument('--resource-lock',type=Path,required=True);args=a.parse_args();base=args.root;lock=json.loads((base/'data_lock.json').read_text());dest=base/'confirmation_lock.json'
 if dest.exists():raise RuntimeError('Confirmation already frozen; do not overwrite')
 if list((base/'response_cache_test').glob('*.parquet')):raise RuntimeError('Test bank already exists before method freeze')
 resource_lock=args.resource_lock.resolve()
 if not resource_lock.is_file():raise RuntimeError('Root resource operating-point lock required before final-test collection')
 resource=json.loads(resource_lock.read_text())
 if not resource.get('operating_points_frozen_before_test',False):raise RuntimeError('Resource lock must assert operating_points_frozen_before_test=true')
 cross_path=ROOT/'experiments/runs/v2/rich_behavior/concurrent_mcq_cross_audit.json'
 cross=json.loads(cross_path.read_text())
 if not cross.get('passed') or not cross.get('before_final_test_outputs'):raise RuntimeError('Contemporary rich/MCQ exposure cross-audit required')
 for item in cross['mcq_inputs']:
  if sha(ROOT/item['path'])!=item['sha256']:raise RuntimeError('Cross-audited MCQ cohort changed')
 cheap=base/'cheap_single_peer_controls';cheap_lock=json.loads((cheap/'selection_lock.json').read_text())
 if not cheap_lock.get('selected_before_test') or cheap_lock['resource_lock_sha256']!=sha(resource_lock):raise RuntimeError('Cheap single-peer controls must be locked against the same resource reference before test')
 if cheap_lock['candidate_freeze_sha256']!=sha(cheap/'candidate_freeze.json'):raise RuntimeError('Cheap-control candidate freeze changed')
 artifacts={os.path.relpath(cross_path,base):sha(cross_path),'data_lock_provenance_supplement.json':sha(base/'data_lock_provenance_supplement.json'),'discovery_followup_ranking_used.csv':sha(base/'discovery_followup_ranking_used.csv'),'fresh_validation_power_check.json':sha(base/'fresh_validation_power_check.json'),'fresh_validation_power_check.csv':sha(base/'fresh_validation_power_check.csv')};count=0
 for p in cheap.rglob('*'):
  if p.is_file():artifacts[str(p.relative_to(base))]=sha(p)
 artifacts[os.path.relpath(ROOT/'experiments/confirmation_single_peer.py',base)]=sha(ROOT/'experiments/confirmation_single_peer.py')
 for run_id in ['campaign']:
  folder=base/'fits'/run_id;globalfile=folder/'all_selections_frozen.json';globalfreeze=json.loads(globalfile.read_text());assert globalfreeze['cells']==len(lock['training_seeds'])*len(lock['settings'])
  artifacts[str(globalfile.relative_to(base))]=sha(globalfile)
  for rel,h in globalfreeze['freeze_hashes'].items():
   f=folder/rel;assert sha(f)==h;freeze=json.loads(f.read_text());artifacts[str(f.relative_to(base))]=h;count+=1
   for filename,digest in freeze['artifacts'].items():
    p=f.parent/filename;assert sha(p)==digest;artifacts[str(p.relative_to(base))]=digest
   for context in freeze['contexts'].values():
    p=(f.parent/context['path']).resolve();assert sha(p)==context['sha256'];artifacts[str(p.relative_to(base))]=context['sha256']
  snap=base/'locked_run_plans';snap.mkdir(exist_ok=True);p=snap/f'{run_id}.json';shutil.copy2(folder/'run_plan.json',p);artifacts[str(p.relative_to(base))]=sha(p)
 for p in [ROOT/'experiments/phase_a_run.py',ROOT/'experiments/adapters_v2.py',ROOT/'experiments/confirmation_collect.py',ROOT/'experiments/confirmation_analyze.py',ROOT/'experiments/mcq_deployment.py',ROOT/'benchmarks/selection_core.py',ROOT/'benchmarks/run_selection.py',resource_lock]:artifacts[os.path.relpath(p,base)]=sha(p)
 final={'protocol_id':lock['protocol_id'],'created_unix':time.time(),'all_fits_and_policies_frozen':True,'resource_operating_points_frozen':True,'resource_lock_path':os.path.relpath(resource_lock,base),'resource_lock_sha256':sha(resource_lock),'test_responses_not_collected':True,'frozen_cells':count,'data_lock_sha256':sha(base/'data_lock.json'),'analysis_code_sha256':sha(ROOT/'experiments/confirmation_analyze.py'),'frozen_artifact_hashes':artifacts,'primary_endpoint':lock['primary_endpoint'],'comparators':lock['comparators'],'familywise_comparisons':lock['familywise_comparisons'],'familywise_alpha':lock['familywise_alpha'],'bootstrap_draws':lock['bootstrap_draws'],'test_collection_allowed_after_this_lock':True}
 dest.write_text(json.dumps(final,indent=2)+'\n');print(json.dumps({k:v for k,v in final.items() if k!='frozen_artifact_hashes'},indent=2));print('hashed_artifacts',len(artifacts))
if __name__=='__main__':main()
