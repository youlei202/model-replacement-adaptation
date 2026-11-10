"""Package a small existing heldout slice for numerical replay, never a new test."""
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from rich_setup import OUT,MODELS,digest,dump
from rich_codec import decode

def main():
 output=OUT/'reproduce/mini_replay';output.mkdir(parents=True,exist_ok=True);lock=json.load(open(OUT/'selection_lock.json'));rlock=json.load(open(OUT/'resources/operating_points_lock.json'));ids=lock['test_item_ids'][:8]
 docs=pd.read_parquet(OUT/'documents.parquet',columns=['item_id','cluster_id','source','context_ids','context_tokens']).set_index('item_id').loc[ids].reset_index();dump(output/'items.json',[{'item_id':r.item_id,'cluster_id':r.cluster_id,'source':r.source,'canonical_token_ids':r.context_ids.tolist(),'context_tokens':int(r.context_tokens)} for r in docs.itertuples()])
 models=[]
 for i,(model,rev,role) in enumerate(MODELS):
  src=OUT/'logits/test'/f'model_{i}.npy';dest=output/f'model_{i}.npy';np.save(dest,np.asarray(np.load(src,mmap_mode='r')[:8]).copy());models.append({'global_id':i,'model_id':model,'revision':rev,'role':role,'file':dest.name,'full_bank_sha256':digest(src),'full_bank_shape':[2048,151936],'slice_shape':[8,151936],'dtype':'float32_centered_logits','slice_sha256':digest(dest)})
 states={}
 for row in lock['choices']+[r for r in rlock['state_cap_sweep_points'] if r['feasible']]:
  sid=row['candidate_id'];blob=(OUT/row['state_path']).read_bytes();decoded=decode(blob);assert decoded['basis_rank']==0,'Minipackagecannotpretendtoincludeomittedpublicbasis'
  if sid not in states:
   dest=output/'states'/f'{sid}.bin';dest.parent.mkdir(exist_ok=True);dest.write_bytes(blob);states[sid]={'state_digest':sid,'file':str(dest.relative_to(output)),'family':decoded['family'],'peer_indices':decoded['peers'],'basis_rank':0,'target_indices':[],'serialized_bytes':len(blob),'source_state_path':row['state_path']}
  if row['target_index'] not in states[sid]['target_indices']:states[sid]['target_indices'].append(row['target_index'])
 losses=pd.read_parquet(OUT/'results/item_losses.parquet');losses=losses[losses.item_id.isin(ids)&losses.candidate_id.isin(states)].copy();allowed={(t,s) for s,r in states.items() for t in r['target_indices']};losses=losses[[ (t,s) in allowed for t,s in zip(losses.target_index,losses.candidate_id) ]];losses.to_csv(output/'recorded_losses.csv',index=False)
 for fn in ['rich_codec.py','rich_inference.py']:shutil.copy2(ROOT/'experiments'/fn,output/fn)
 shutil.copy2(OUT/'reports/replay_example_validation.json',output/'cross_device_precision_diagnostic.json')
 dump(output/'manifest.json',{'scope':'Eightpre-existingtestdocumentsforcodec/predictionarithmeticreplay, not independentconfirmation orfulltestreproduction','test_slice_indices':list(range(8)),'selection_lock_sha256':digest(OUT/'selection_lock.json'),'models':models,'states':list(states.values()),'raw_full_banks_excluded':True,'public_bases_for_nonselected_residual_candidates_excluded':True,'primary_selected_states_need_public_basis':False,'runtime':'CPU torch+numpy; sameB200officialmetrics maydifferbyfloatingpointreduction; strict1e-6cross-devicecheckobserved6.97e-6andretained','source_numeric_stack':'BF16 modelinference microbatch1 onB200; storedFP32centeredfulllogits; full151936outputslots','hashes':{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file() and p.name!='manifest.json'}})
 print(output,'files',sum(1 for p in output.rglob('*') if p.is_file()),'bytes',sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
if __name__=='__main__':main()
