"""Inference-only full-vocabulary replacement math; no fitting or target/model loading."""
import torch

def predict(state,peers,basis=None):
 q=len(peers);fam=state['family'];a=[torch.as_tensor(x,device=peers[0].device) for x in state['arrays']]
 if fam=='probability_mean':return torch.stack([torch.softmax(x,dim=-1) for x in peers]).mean(0),True
 if fam=='logit_mean':return sum(peers)/q,False
 pred=sum(w*x for w,x in zip(a[0],peers))
 if fam=='vocabulary_bias':pred=pred+a[1]
 if fam in ('ridge_residual','factored_residual'):
  r=state['basis_rank'];u=basis[:,:r];features=torch.cat([x@u for x in peers]+[torch.ones((len(peers[0]),1),device=peers[0].device)],dim=1)
  correction=features@a[1].T if fam=='ridge_residual' else (features@a[2].T)@a[1].T
  pred=pred+correction@u.T
 return pred,False

def metrics(target,pred,is_prob=False):
 lp=torch.log_softmax(target,dim=-1);p=lp.exp()
 if is_prob:q=pred; lq=pred.clamp_min(torch.finfo(torch.float32).tiny).log()
 else:lq=torch.log_softmax(pred,dim=-1);q=lq.exp()
 top=torch.topk(p,k=2,dim=-1).values
 return {'tv':.5*(p-q).abs().sum(-1),'kl':(p*(lp-lq)).sum(-1).clamp_min(0),'argmax_disagreement':(target.argmax(-1)!=q.argmax(-1)).float(),'target_argmax':target.argmax(-1),'prediction_argmax':q.argmax(-1),'target_margin':top[:,0]-top[:,1]}
