"""Add explicit scientific-contract columns to copies of already-frozen rich results."""
import hashlib,json,struct,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from rich_setup import OUT,MODELS,digest,dump
from rich_codec import HEADER,ARRAY

def meaningful_bits(blob):
 h=HEADER.unpack_from(blob);off=HEADER.size;padding=0
 for _ in range(h[-2]):
  rows,cols,n,nb,scale=ARRAY.unpack_from(blob,off);off+=ARRAY.size+nb
  if h[4]==3 and n%2:padding+=4
 assert off==len(blob)
 return len(blob)*8-padding

def main():
 root=OUT/'reports/contract';root.mkdir(parents=True,exist_ok=True);lock=json.load(open(OUT/'selection_lock.json'));result=OUT/'results';sources=json.load(open(OUT/'sources.json'));datasetrev=json.dumps({r['id']:r['revision'] for r in sources},sort_keys=True)
 rows=pd.read_csv(result/'all_pairs_by_budget.csv');rows['protocol_id']='replacement_v2_rich_common_context_v1';rows['data_role']='external_confirmation_test';rows['dataset_revision']=datasetrev;rows['target_id']=rows.target_index.map(lambda i:MODELS[int(i)][0]);rows['target_revision']=rows.target_index.map(lambda i:MODELS[int(i)][1]);rows['peer_ids']=rows.peers.map(lambda p:json.dumps([MODELS[int(i)][0] for i in str(p).split(',')]));rows['peer_revisions']=rows.peers.map(lambda p:json.dumps([MODELS[int(i)][1] for i in str(p).split(',')]));rows['q']=2;rows['training_seed']=rows.peers.map(lambda p:81724033+10*int(str(p).split(',')[0])+int(str(p).split(',')[1]));rows['budget_bits']=(rows.cap_bytes*8).where(rows.cap_bytes.gt(0));rows['budget_kind']=np.where(rows.cap_bytes.gt(0),'finite_cap','uncapped_or_direct_control');rows['actual_bits']=rows.actual_bytes*8;rows['meaningful_bits']=[meaningful_bits((OUT/p).read_bytes()) for p in rows.state_path];rows['adapter_family']=rows.family;rows['state_digest']=rows.candidate_id;rows['validation_metric']=rows.validation_tv;rows['selection_policy']=rows.scope.map(lambda s:'per_pair_validation_'+s);rows['selected_before_test']=True;rows['test_metric']=rows.test_tv;rows['test_cluster_count']=lock['actual_test_clusters'];rows['error_failure_status']='success';rows.to_csv(root/'all_candidates_by_budget.csv',index=False)
 losses=pd.read_parquet(result/'policy_item_losses.parquet');docs=pd.read_parquet(OUT/'documents.parquet',columns=['item_id','context_ids']);hashes={r.item_id:hashlib.sha256(np.asarray(r.context_ids,dtype='<i8').tobytes()).hexdigest() for r in docs.itertuples()};losses['protocol_id']='replacement_v2_rich_common_context_v1';losses['data_role']='external_confirmation_test';losses['task_id']='rich_next_token';losses['target_id']=losses.target_index.map(lambda i:MODELS[int(i)][0]);losses['prompt_hash']=losses.item_id.map(hashes);losses['seed']='rich-natural-docs-v1';losses['budget_bits']=losses.cap_bytes*8;losses['state_digest']=losses.candidate_id;losses.to_parquet(root/'per_query_policy_losses.parquet',index=False)
 note={'source_selection_lock_sha256':digest(OUT/'selection_lock.json'),'source_evaluation_marker_sha256':digest(result/'evaluation_complete.json'),'copies_only_no_refitting_or_reselection':True,'candidate_table_scope':'Allretainedvalidationenvelopes perpair/family/rank/cap, includingunbounded/directcontrols; thefull2196candidatevalidationledger remainsunderfits/*/validation_candidates.parquet; nopretendallrawcandidateshadtestselection.','meaningful_bits':'Literal encoded bits less finalhalfbyte padding inodd int4arrays;headers/scales/modelmaskallcounted','prompt_hash':'SHA256ofthecanonicalcontexttokenIDarrayencodedlittleendianint64, identicalIDsforallmodels','training_seed':'PeerbasisSVDseed81724033+10*globalpeer0+globalpeer1; targetindependent, calibrationdeterministic','raw_full_probability_predictions':'Notduplicatedforeachstate; exactlyreconstructablefromstate+minimal/publicbasis+rawfullpeerlogits via rich_replay.py. Actualpredictedargmaxisretainedperquery.','zero_policy_variance':'Allfourregisteredriskfixedandjoint chooseidenticalpair/states atallcaps forboth targets. Zeroobservedgainisidentityofpolicies; power_n2fromSD0isnot evidenceforpositivegain. 2048freshtestclustersstillusedforabsolutequality/resourceoperatingpoints.','code_sha256':digest(Path(__file__))};dump(root/'data_dictionary.json',note)
 print('exported',len(rows),'candidatebudgetrows',len(losses),'perquerypolicyrows',flush=True)
if __name__=='__main__':main()
