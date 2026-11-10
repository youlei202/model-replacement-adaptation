"""Small numerical integration checks for serialized rich predictors and full support."""
import sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT/'experiments/experiment_config'))
from rich_codec import encode,decode,self_test
from rich_fit import predict,metrics
from reference.full_vocab import exact_metrics
self_test();rng=np.random.default_rng(5);vocab=1001;rank=16;n=9;q=2
peers=[torch.tensor(rng.normal(size=(n,vocab)).astype('float32')) for _ in range(q)];basis=torch.tensor(np.linalg.qr(rng.normal(size=(vocab,rank)))[0].astype('float32'));w=np.array([.3,1.2],dtype='float32');A=rng.normal(size=(rank,q*rank+1)).astype('float32')*.01
blob=encode('ridge_residual',[0,2],rank,[w,A],vocab=vocab);state=decode(blob);actual,isp=predict(state,peers,basis);x=torch.cat([p@basis for p in peers]+[torch.ones((n,1))],1);expected=w[0]*peers[0]+w[1]*peers[1]+(x@torch.tensor(A).T)@basis.T
assert torch.equal(actual,expected) and not isp
assert state['arrays'][0].sum()!=1,'NNLS mustnotbesimplexnormalized'
target=peers[0];m=metrics(target,actual);reference=exact_metrics(target.numpy(),actual.numpy());assert np.max(np.abs(m['tv'].numpy()-reference['tv']))<2e-7
L=rng.normal(size=(rank,4)).astype('float32');R=rng.normal(size=(4,rank*q+1)).astype('float32');f=decode(encode('factored_residual',[0,2],rank,[w,L,R],vocab=vocab));p,_=predict(f,peers,basis);e=w[0]*peers[0]+w[1]*peers[1]+((x@torch.tensor(R).T)@torch.tensor(L).T)@basis.T;assert torch.equal(p,e)
try:decode(blob+b'x')
except ValueError:pass
else:raise AssertionError('codecacceptedtrailingbytes')
try:encode('nnls',[2,0],0,[w])
except ValueError:pass
else:raise AssertionError('codecacceptedunsortedpeerIDs')
print('Rich6numericalintegrationchecks passed: native residual/factor replay, non-simplexNNLS, exactTV, malformedcodec/order rejected')
