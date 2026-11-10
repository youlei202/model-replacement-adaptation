"""Replay a saved rich predictor without target weights or target response files.

Example: python experiments/rich_replay.py --state STATE.bin --role test
         --start 0 --count 8 --output replay_probabilities.npy
"""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from rich_setup import OUT
from rich_codec import decode
from rich_inference import predict

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True);p.add_argument('--role',choices=['fit','validation','test'],default='test');p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=8);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=decode(a.state.read_bytes());assert set(s['peers'])<=set([0,1,2]),'Retiredtargetcannotbepeer';stop=a.start+a.count
 peers=[torch.as_tensor(np.asarray(np.load(OUT/'logits'/a.role/f'model_{i}.npy',mmap_mode='r')[a.start:stop]).copy()) for i in s['peers']];basis=None
 if s['basis_rank']:
  pair=s['peers'];basis=torch.as_tensor(np.load(OUT/'public_contexts'/f'pair_{pair[0]}_{pair[1]}.npy')[:,:s['basis_rank']].copy())
 with torch.inference_mode():pred,prob=predict(s,peers,basis);out=pred if prob else torch.softmax(pred,dim=-1)
 a.output.parent.mkdir(parents=True,exist_ok=True);np.save(a.output,out.numpy());print(json.dumps({'shape':list(out.shape),'target_weights_or_responses_loaded':False,'state_bytes':len(a.state.read_bytes()),'peer_indices':s['peers'],'output':str(a.output)}))
if __name__=='__main__':main()
