"""Reproduce rich quality/resource tables from frozen artifacts only."""
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT/'experiments/experiment_config'))
from rich_setup import OUT,digest,dump
from reference.statistics import paired_cluster_interval

def main():
 result=OUT/'results';resource=OUT/'resources';reports=OUT/'reports';reports.mkdir(exist_ok=True)
 score=pd.read_csv(result/'policy_scores.csv');primary=pd.read_csv(result/'primary_confirmation.csv');loss=pd.read_parquet(result/'item_losses.parquet');lock=json.load(open(OUT/'selection_lock.json'));rlock=json.load(open(resource/'operating_points_lock.json'));points=rlock['points'];table=[];lookup={(t,s):g.sort_values('item_id') for (t,s),g in loss.groupby(['target_index','candidate_id'])}
 for row in points:
  entry={k:row.get(k) for k in ['target_index','policy','margin_tv','feasible','candidate_id','quality_reference_state','failure_status']}
  if row['feasible']:
   a=lookup[(row['target_index'],row['candidate_id'])];b=lookup[(row['target_index'],row['quality_reference_state'])];assert a.item_id.tolist()==b.item_id.tolist();d=a.tv.to_numpy()-b.tv.to_numpy();comparisons=12 if row['margin_tv']==.005 else 48;ci=paired_cluster_interval(d,a.cluster_id,comparisons=comparisons,draws=10000,seed=81724038)
   entry.update({'test_tv':a.tv.mean(),'reference_test_tv':b.tv.mean(),'test_argmax_disagreement':a.argmax_disagreement.mean(),'quality_gap_tv':ci.estimate,'quality_gap_simultaneous_low':ci.low,'quality_gap_simultaneous_high':ci.high,'heldout_noninferiority_confirmed':ci.high<=row['margin_tv'],'simultaneous_comparisons':comparisons,'test_clusters':len(a),'actual_state_bytes':row['actual_bytes']})
  table.append(entry)
 quality=pd.DataFrame(table);quality.to_csv(reports/'matched_fidelity_quality.csv',index=False)
 sweep_rows=[];ci_cache={}
 for row in rlock['state_cap_sweep_points']:
  entry={k:row.get(k) for k in ['target_index','policy','margin_tv','resource_state_cap_bytes','feasible','candidate_id','quality_reference_state','failure_status']}
  if row['feasible']:
   key=(row['target_index'],row['candidate_id'],row['quality_reference_state'])
   a=lookup[key[:2]];b=lookup[(key[0],key[2])]
   if key not in ci_cache:ci_cache[key]=paired_cluster_interval(a.tv.to_numpy()-b.tv.to_numpy(),a.cluster_id,comparisons=384,draws=10000,seed=81724042)
   ci=ci_cache[key];entry.update({'test_tv':a.tv.mean(),'quality_gap_tv':ci.estimate,'quality_gap_simultaneous_high':ci.high,'heldout_noninferiority_confirmed':ci.high<=row['margin_tv'],'development_GPU_ms':row['development_GPU_ms'],'actual_state_bytes':row['actual_bytes']})
  sweep_rows.append(entry)
 pd.DataFrame(sweep_rows).to_csv(reports/'matched_fidelity_state_cap_sweep.csv',index=False)
 policies=pd.read_parquet(result/'policy_item_losses.parquet');policies['target_margin_stratum']=pd.cut(policies.target_margin,[-1e-9,.1,.5,1.0000001],labels=['low_0_to_.1','middle_.1_to_.5','high_.5_to_1']);policies.groupby(['target_index','policy','cap_bytes','source','target_margin_stratum'],observed=True).agg(clusters=('item_id','size'),mean_tv=('tv','mean'),mean_kl=('kl','mean'),argmax_disagreement=('argmax_disagreement','mean')).reset_index().to_csv(reports/'source_margin_strata.csv',index=False)
 reduction=[]
 if (resource/'benchmark_complete.json').exists():
  timings=pd.read_csv(resource/'timing_raw.csv');block=timings.groupby(['system_digest','batch_size','context_tokens','timing_block']).agg(GPU_ms=('GPU_ms','mean'),wall_ms=('end_to_end_ms','mean')).reset_index();rng=np.random.default_rng(81724039)
  for target in [3,4]:
   for margin in [0,.0025,.005,.01]:
    relevant=[r for r in points if r['target_index']==target and r['margin_tv']==margin];joint=next(r for r in relevant if r['policy']=='budget_aware')
    for fixed in relevant:
     if fixed['policy']=='budget_aware':continue
     for batch in [1,8]:
      for length in [0,128,512]:
       out={'target_index':target,'fixed_policy':fixed['policy'],'margin_tv':margin,'batch_size':batch,'context_tokens':length,'joint_feasible':joint['feasible'],'fixed_feasible':fixed['feasible']}
       if joint['feasible'] and fixed['feasible']:
        a=block[(block.system_digest==joint['candidate_id'])&(block.batch_size==batch)&(block.context_tokens==length)].sort_values('timing_block');b=block[(block.system_digest==fixed['candidate_id'])&(block.batch_size==batch)&(block.context_tokens==length)].sort_values('timing_block');assert a.timing_block.tolist()==b.timing_block.tolist() and len(a)>=10
        idx=rng.integers(len(a),size=(10000,len(a)));av,bv=a.GPU_ms.to_numpy(),b.GPU_ms.to_numpy();boot=1-av[idx].mean(1)/bv[idx].mean(1);comps=10 if margin==.005 and batch==1 and length==0 else 240;lo,hi=np.quantile(boot,[.05/(2*comps),1-.05/(2*comps)])
        out.update({'joint_GPU_ms_per1000contexts':av.mean()*1000/batch,'fixed_GPU_ms_per1000contexts':bv.mean()*1000/batch,'mean_relative_resource_reduction':1-av.mean()/bv.mean(),'simultaneous_low':lo,'simultaneous_high':hi,'independent_timing_blocks':len(a),'comparisons':comps,'design20percent_reached':1-av.mean()/bv.mean()>=.2,'positive_resource_reduction_confirmed':lo>0,'joint_system_digest':joint['candidate_id'],'fixed_system_digest':fixed['candidate_id']})
       reduction.append(out)
  pd.DataFrame(reduction).to_csv(reports/'resource_reduction.csv',index=False)
  # Matched nominal resourcecaps from observed deployment peaks, fixedbeforequalitytest via developmentprofiles.
 direct_quality=[]
 from scipy.special import softmax
 for target in [3,4]:
  t=np.load(OUT/'logits/test'/f'model_{target}.npy',mmap_mode='r')
  for peer in [0,1,2]:
   p=np.load(OUT/'logits/test'/f'model_{peer}.npy',mmap_mode='r');tv=[];arg=[]
   for start in range(0,len(t),16):
    a=np.asarray(t[start:start+16],dtype=np.float64);b=np.asarray(p[start:start+16],dtype=np.float64);tv.extend((.5*np.abs(softmax(a,axis=1)-softmax(b,axis=1)).sum(1)).tolist());arg.extend((a.argmax(1)!=b.argmax(1)).tolist())
   direct_quality.append({'target_index':target,'peer_index':peer,'direct_peer_test_TV':float(np.mean(tv)),'direct_peer_argmax_disagreement':float(np.mean(arg)),'fresh_test_clusters':len(t),'selection_role':'predeterminedpeercontextreference_nottestselected'})
 pd.DataFrame(direct_quality).to_csv(reports/'direct_small_peer_quality.csv',index=False)
 text=['# Rich fixed-context replacement evidence','',f'Five compatible checkpoints; retired targets Qwen3-8B and Qwen3-14B, callable peers Qwen3-0.6B/1.7B/4B. All three q=2 subsets included. {lock["actual_test_clusters"]} fresh document clusters; two sources weighted equally.','', 'Full-vocabulary support has151936 output slots, including267 shared reserved unmapped slots. This is conditional next-token fidelity on common natural contexts, not free-running generation.','', '## Primary fresh policy contrasts','',primary.to_markdown(index=False),'','## Validation-frozen matched-fidelity operating points','',quality.to_markdown(index=False),'']
 if reduction:text.extend(['## Measured complete-pipeline resource comparisons','',pd.DataFrame(reduction).query('margin_tv == 0.005 and batch_size == 1 and context_tokens == 0').to_markdown(index=False),''])
 text.extend(['## Scope and accounting','', 'The additional cost-aware fixedpair is chosenonce acrossstatecaps beforetest; savingsversusrisk-selectedfixedsetsalone do not establish consequentialstatecoupling. All fixed policies use exactly the joint policy\'s adapter menu and tune independently at every cap. Direct peers, NNLS, convex, temperature, full vocabulary bias, ridge and factored residuals remain in the envelope. Target-derived scalar/matrix/bias state is serialized by the RBV2 codec; public bases use only the selected peers on fit documents.','', 'Deployment measurement includes actual peer forward passes, state decoding/transfers, projection and full probability conversion. CUDA event elapsed time is distinguished from wall latency and is not a claim of summed kernel-busy time. Primarycost uses untruncatedcanonicalnaturalcontexts fromthesamepopulationasquality (context_tokens=0); controlled128/512timings aresecondary. Logicalbatch8 iterates8canonicalmicrobatch1forwards, notfusedGPUbatch8. Warm measurements use10 counterbalanced randomized blocks and200 requests per setting; initial loading is reported separately with warm filesystem cache.','', 'The registered two retired targets are distinct, but each timed process serves one fitted target system. This is not an observed multi-tenant-capacity result. Conditional accounting lists peer infrastructure separately; standalone accounting includes peer weights and actual minimal basis files plus target state.','', 'Dataset/checkpoint access was public, but this does not establish absence from foundation-model pretraining. The arXiv dataset card gives no blanket redistribution license; reproduce by pinned acquisition, and do not redistribute raw full articles automatically. WikiText card YAML/body license versions differ and both are retained in source provenance.','', 'No empirical kernel spectrum is claimed to verify the theoretical Gaussian covariance mechanism. Zero/negative policy gains and infeasible fixed operating points are preserved.'])
 (reports/'RICH_EVIDENCE.md').write_text('\n'.join(text)+'\n')
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  pairs=pd.read_csv(result/'all_pairs_by_budget.csv');fig,axes=plt.subplots(1,2,figsize=(12,4))
  for target,ax in zip([3,4],axes):
   for peers,g in pairs[(pairs.target_index==target)&(pairs.scope=='all')].groupby('peers'):
    g=g.sort_values('cap_bytes');ax.plot(g.cap_bytes,g.test_tv,'o-',label='peers '+str(peers))
   ax.set_xscale('log',base=2);ax.set_xlabel('Retained state cap(bytes)');ax.set_ylabel('Fresh full-support TV');ax.set_title(['Qwen3-8B','Qwen3-14B'][target-3]);ax.legend()
  fig.tight_layout();fig.savefig(reports/'rich_error_state_curves.png',dpi=180);plt.close(fig)
 except ImportError:pass
 dump(reports/'report_manifest.json',{'selection_lock_sha256':digest(OUT/'selection_lock.json'),'evaluation_complete_sha256':digest(result/'evaluation_complete.json'),'resource_complete':(resource/'benchmark_complete.json').exists(),'code_sha256':digest(Path(__file__)),'outputs':{p.name:digest(p) for p in reports.iterdir() if p.is_file() and p.name!='report_manifest.json'}})
 print(primary.to_string(index=False))
if __name__=='__main__':main()
