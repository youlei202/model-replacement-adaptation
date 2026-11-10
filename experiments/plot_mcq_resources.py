#!/usr/bin/env python3
"""Scientific plots of frozen operating points, fresh quality, and live resources.

No test-optimal Pareto envelope is selected. Every plotted operating point was
chosen from validation and development timing; zero/negative contrasts remain.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
LABELS={'budget_aware':'Joint','fixed_native_full':'Fixed native full','fixed_unbounded_best':'Fixed unbounded best','fixed_mean_budget':'Fixed mean budget','fixed_minimax_validation_regret':'Fixed minimax regret','fixed_cost_aware_mean_cap':'Fixed cost-aware (diagnostic)'}
COLORS={'budget_aware':'black','fixed_native_full':'#8C6D31','fixed_unbounded_best':'#756BB1','fixed_mean_budget':'#D95F0E','fixed_minimax_validation_regret':'#238B45','fixed_cost_aware_mean_cap':'#2171B5'}
MARKERS={'budget_aware':'o','fixed_native_full':'s','fixed_unbounded_best':'^','fixed_mean_budget':'D','fixed_minimax_validation_regret':'v','fixed_cost_aware_mean_cap':'X'}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def truth(v):return str(v).lower() in ('true','1','1.0')
def save(fig,out,name):
 for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();out=a.output or a.analysis/'figures';out.mkdir(parents=True,exist_ok=True)
 required=['fresh_operating_point_quality.csv','resource_summary.csv','paired_resource_contrasts.csv','timing_blocks.csv']
 for name in required:
  if not(a.analysis/name).exists():raise SystemExit(f'Measured input missing: {name}; no figures fabricated')
 quality=pd.read_csv(a.analysis/required[0]);summary=pd.read_csv(a.analysis/required[1]);contrasts=pd.read_csv(a.analysis/required[2]);blocks=pd.read_csv(a.analysis/required[3]);catalog=json.loads((a.catalog/'candidate_catalog.json').read_text());names={r['index']:r['model_id'] for r in catalog['model_registry']};targets=sorted(quality.target_index.unique())
 natural=summary[(summary.batch_size==1)&(summary.context_tokens==0)].set_index('system_id')
 fig,axes=plt.subplots(1,len(targets),figsize=(6.4*len(targets),4.8),squeeze=False,constrained_layout=True)
 for ax,target in zip(axes[0],targets):
  sub=quality[quality.target_index==target];infeasible=int((sub.status!='eligible').sum())
  for policy in LABELS:
   d=sub[(sub.policy==policy)&(sub.status=='eligible')].sort_values('quality_margin');xs=[];ys=[]
   for r in d.to_dict('records'):
    if r['selected_system'] not in natural.index:raise ValueError('Frozen system lacks natural live measurement')
    cost=float(natural.loc[r['selected_system'],'mean_end_to_end_ms_per1000'])/1000;x=float(r['quality_gap_TV']);primary=np.isclose(r['quality_margin'],.005)
    filled=truth(r.get('noninferiority_pass_primary_adjustment',False));size=70 if primary else 25
    ax.hlines(cost,r['simultaneous_low'],r['simultaneous_high'],color=COLORS[policy],alpha=.75 if primary else .25,lw=1.4 if primary else .7)
    ax.scatter([x],[cost],marker=MARKERS[policy],s=size,facecolors=COLORS[policy] if filled else 'none',edgecolors=COLORS[policy],alpha=1 if primary else .5,zorder=3);xs.append(x);ys.append(cost)
   if len(xs):ax.plot(xs,ys,color=COLORS[policy],lw=.6,alpha=.35,label=LABELS[policy])
  ax.axvline(0,color='.6',lw=.8);ax.axvline(.005,color='black',ls='--',lw=.8,label='Primary TV tolerance .005');ax.set_title(names[target]);ax.set_xlabel('Fresh TV difference from frozen reference');ax.set_ylabel('Coordinate-output time (ms/context)');ax.grid(alpha=.2)
  ax.text(.01,.01,f'{infeasible} infeasible frozen points retained in table',transform=ax.transAxes,fontsize=8,va='bottom')
 axes[0,0].legend(fontsize=7);fig.suptitle('Validation-frozen operating points: fresh fidelity and measured coordinate-output time\nLarge markers: primary .005 margin; filled: TV and argmax noninferiority; other margins/extra fixed policy descriptive',fontsize=10)
 save(fig,out,'mcq_cost_fidelity_operating_points')
 fig,axes=plt.subplots(1,len(targets),figsize=(6.4*len(targets),4.5),squeeze=False,constrained_layout=True)
 fixed=[k for k in LABELS if k!='budget_aware'];labels=[LABELS[p] for p in fixed]
 for ax,target in zip(axes[0],targets):
  sub=contrasts[(contrasts.target_index==target)&np.isclose(contrasts.quality_margin,.005)]
  for i,policy in enumerate(fixed):
   d=sub[sub.comparator==policy]
   if d.empty or d.iloc[0].status!='measured':ax.text(0,i,'infeasible',va='center',fontsize=8);continue
   r=d.iloc[0];lo,hi=float(r.resource_reduction_low),float(r.resource_reduction_high);value=float(r.resource_reduction)
   passed=truth(r.get('joint_fresh_TV_and_argmax_NI',False)) and truth(r.get('fixed_fresh_TV_and_argmax_NI',False));color=COLORS[policy] if passed else '.55'
   ax.hlines(i,lo,hi,color=color,lw=1.5);ax.scatter(value,i,marker=MARKERS[policy],s=55,facecolors=color if passed else 'none',edgecolors=color,zorder=3)
   if not passed:ax.annotate('fidelity not confirmed',(value,i),xytext=(4,7),textcoords='offset points',fontsize=6,color='.4')
  ax.axvline(0,color='black',lw=1);ax.axvline(.2,color='.4',ls=':',lw=1);ax.set_yticks(range(len(fixed)),labels);ax.invert_yaxis();ax.set_title(names[target]);ax.set_xlabel('1 − joint time / fixed time');ax.grid(axis='x',alpha=.2)
 fig.suptitle('Paired coordinate-endpoint time reductions at the primary fidelity tolerance\nIntervals use independent timing blocks; the additional cost-aware fixed policy is diagnostic',fontsize=10)
 save(fig,out,'mcq_paired_resource_reductions')
 # Shared system identity prevents counting duplicate policy labels as tenants.
 selected=set(quality.loc[np.isclose(quality.quality_margin,.005)&quality.status.eq('eligible'),'selected_system']);selected.update(quality.reference_system.dropna())
 mem=natural.loc[sorted(selected)].copy();mem.to_csv(out/'plotted_memory_accounting.csv')
 fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True);x=np.arange(len(mem));labels=[s[:3]+'…'+s[-6:] for s in mem.index]
 axes[0].bar(x,mem.model_parameter_and_buffer_bytes/2**30,label='Peer weights/buffers',color='#9ECAE1');axes[0].bar(x,mem.shared_deployment_resident_bytes/2**30,bottom=mem.model_parameter_and_buffer_bytes/2**30,label='Shared public context',color='#3182BD');axes[0].set_ylabel('Accounted weights + shared arrays (GiB)');axes[0].legend(fontsize=7)
 width=.36;axes[1].bar(x-width/2,mem.state_binary_bytes,width,label='Serialized state',color='#FDAE6B');axes[1].bar(x+width/2,mem.resident_target_state_total_bytes,width,label='Encoded bytes + decoded numeric payload',color='#E6550D');axes[1].set_ylabel('Target state (bytes)');axes[1].legend(fontsize=7)
 axes[2].bar(x,mem.peak_allocated_VRAM_bytes/2**30,color='#74C476',label='Measured peak allocated VRAM');axes[2].plot(x,mem.peak_reserved_VRAM_bytes/2**30,'k_',markersize=12,label='Peak reserved VRAM');axes[2].set_ylabel('Measured process VRAM (GiB)');axes[2].legend(fontsize=7)
 for ax in axes:ax.set_xticks(x,labels,rotation=60,ha='right');ax.grid(axis='y',alpha=.2)
 fig.suptitle('Separate peer/shared infrastructure, retained target state, and measured runtime memory\nEach unique deployed system counted once. Object overhead is separate in resident_state_supplement.json and process RSS; no simultaneous retired-tenant claim.',fontsize=10)
 save(fig,out,'mcq_memory_accounting')
 b=blocks[(blocks.batch_size==1)&(blocks.context_tokens==0)&blocks.system_id.isin(selected)]
 fig,ax=plt.subplots(figsize=(10,4.6),constrained_layout=True)
 for sid,d in b.groupby('system_id'):
  d=d.sort_values('timing_block');ax.plot(d.timing_block,d.mean_end_to_end_ms_per1000/1000,'o-',lw=.9,ms=3,label=sid[:3]+'…'+sid[-6:])
 ax.set_xlabel('Paired timing block');ax.set_ylabel('Mean coordinate-output time (ms/context)');ax.set_title('Natural contexts, batch1: block variability for every frozen primary system');ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2)
 save(fig,out,'mcq_timing_block_variability')
 manifest={'selection':'All points are frozen from validation/development timing; no test-optimal envelope','fidelity_scope':'Batch1 natural untruncated only; controlled128/512 and fusedbatch8 are not assigned this quality result','inputs':{name:digest(a.analysis/name) for name in required},'timing_endpoint':'Tokenization and live peer inference through adapter Helmert coordinates; excludes inverse transform/probability normalization. Adapter decode/load occurs before warm requests.','catalog_sha256':digest(a.catalog/'candidate_catalog.json'),'outputs':{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='figure_manifest.json'}}
 (out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'output':str(out),'figures':4,'negative_zero_results_retained':True}))
if __name__=='__main__':main()
