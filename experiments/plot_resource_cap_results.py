#!/usr/bin/env python3
"""Static reporting plots for both new measured-cap cohorts; no reselection."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

FIXED=['fixed_native_full','fixed_unbounded_best','fixed_mean_budget','fixed_minimax_validation_regret']
LABELS={'fixed_native_full':'native full fixed','fixed_unbounded_best':'unbounded best fixed',
        'fixed_mean_budget':'mean-cap fixed','fixed_minimax_validation_regret':'minimax fixed',
        'fixed_cost_aware_mean_cap':'cost-aware fixed [diagnostic]',
        'fixed_resource_mean_frontier':'cost-aware fixed [diagnostic]',
        'budget_aware_nnls':'joint NNLS [diagnostic]','joint_cheap':'joint cheap [diagnostic]',
        'budget_aware':'joint','joint':'joint'}
NAMES={('MCQ',1):'Olmo-3-7B',('MCQ',3):'Qwen3.5-9B',
       ('Rich',3):'Qwen3-8B',('Rich',4):'Qwen3-14B'}


def truth(v):return str(v).lower() in ('true','1','1.0')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output',type=Path);args=parser.parse_args();run=args.run_root
    out=args.output or run/'reports/figures';out.mkdir(parents=True,exist_ok=True)
    sources=[];routes=[]
    for route,folder,marker,result,cov in [
        ('MCQ','resource_cap_confirmation','complete.json','guarded_confirmation_results.csv','coverage_and_cap_compliance.csv'),
        ('Rich','rich_behavior/matched_resource_supplement','supplement_complete.json','primary_confirmation.csv','policy_quality_resources.csv')]:
        base=run/folder
        assert json.loads((base/marker).read_text())['complete'] is True
        source=base/result;coverage=base/cov;sources.extend([base/marker,source,coverage])
        data=pd.read_csv(source)
        if route=='Rich':data=data.rename(columns={'fixed_policy':'comparator','cap_averaged_gain_TV':'gain_tv',
            'simultaneous_low':'simultaneous_ci_low','simultaneous_high':'simultaneous_ci_high','fresh_clusters':'independent_test_clusters'})
        assert len(data)==12
        routes.append((route,data,pd.read_csv(coverage)))
    bounds_path=run/'reports/resource_cap_probability_guard/system_latency_guard.csv';sources.append(bounds_path)
    bounds=pd.read_csv(bounds_path)
    probability_path=run/'resource_cap_confirmation/probability_endpoint_audit/resource_summary.csv';sources.append(probability_path)
    probability=pd.read_csv(probability_path).set_index('system_id')
    colors=['#2166ac','#d97706'];records=[]
    fig,axes=plt.subplots(1,2,figsize=(16,7),constrained_layout=True)
    for ax,(route,data,_) in zip(axes,routes):
        labels=[]
        for target,g in data.groupby('target_index',sort=True):
            ordering=FIXED+[p for p in g.comparator if p not in FIXED]
            for policy in ordering:
                r=g[g.comparator.eq(policy)].iloc[0];y=len(labels);primary=policy in FIXED
                labels.append(f"{NAMES[route,int(target)]} / {LABELS.get(policy,policy)} ({int(r.number_common_caps)} caps)")
                ax.hlines(y,r.simultaneous_ci_low,r.simultaneous_ci_high,color=colors[not primary],linewidth=2)
                ax.plot(r.gain_tv,y,'o' if primary else 's',color=colors[not primary],markersize=5)
                records.append({'route':route,'target_index':target,'comparator':policy,'primary':primary,
                    'gain_tv':r.gain_tv,'CI_low':r.simultaneous_ci_low,'CI_high':r.simultaneous_ci_high,
                    'common_caps':r.number_common_caps,'test_clusters':r.independent_test_clusters})
        ax.axvline(0,color='black',linewidth=.8);ax.axvline(.005,color='#b2182b',linestyle='--',linewidth=1)
        lo=min(-.002,float(data.simultaneous_ci_low.min())-.001);hi=max(.006,float(data.simultaneous_ci_high.max())+.001)
        ax.set_xlim(lo,hi);ax.set_yticks(range(len(labels)),labels,fontsize=8);ax.invert_yaxis()
        ax.grid(axis='x',alpha=.2);ax.set_xlabel('TV gain: comparator − joint; within-query cap average')
        ax.set_title(f"{route}: {int(data.independent_test_clusters.iloc[0]):,} new clusters",fontsize=12)
    fig.suptitle('Later confirmation under measured latency / allocated-VRAM caps\n'
        'Original fixed comparisons: simultaneous 95% family-16 intervals; dashed line: .005 design effect',fontsize=12)
    fig.supxlabel('Orange controls are diagnostic: MCQ uses family-16 bounds; rich uses family-4 bounds. '
        'A positive quality interval alone does not establish fresh resource-cap compliance.',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(out/f'resource_cap_confirmation_gains.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig);pd.DataFrame(records).to_csv(out/'resource_cap_confirmation_gains.csv',index=False)

    palette=['#d9d9d9','#d95f59','#f4bd55','#439b77'];records=[]
    fig,axes=plt.subplots(2,2,figsize=(17,8.5),constrained_layout=True)
    for column,(route,_,coverage) in enumerate(routes):
        route_bounds=bounds[bounds.route.str.startswith(route)].set_index('system_id')
        for row,(target,g) in enumerate(coverage.groupby('target_index',sort=True)):
            ax=axes[row,column];joint='budget_aware' if route=='MCQ' else 'joint'
            policies=[joint]+FIXED+[p for p in g.policy.unique() if p not in FIXED+[joint]]
            caps=list(dict.fromkeys(g.cap_id));matrix=np.full((len(policies),len(caps)),np.nan);caplabels=[]
            for ci,cap in enumerate(caps):
                first=g[g.cap_id.eq(cap)].iloc[0]
                lat=first['latency_cap_ms_per_context' if route=='MCQ' else 'mean_wall_latency_cap_ms']
                caplabels.append(f"{lat:g}\n{first.allocated_VRAM_cap_bytes/2**30:g}")
                for pi,policy in enumerate(policies):
                    r=g[g.cap_id.eq(cap)&g.policy.eq(policy)].iloc[0]
                    feasible=(r.status=='eligible') if route=='MCQ' else truth(r.feasible_on_development)
                    state=0
                    if feasible:
                        sid=r['system_id' if route=='MCQ' else 'candidate_id']
                        mean=r['final_mean_ms_per_context' if route=='MCQ' else 'mean_wall_ms']
                        memory=r['final_peak_allocated_bytes' if route=='MCQ' else 'peak_allocated_bytes']
                        point=truth(r.final_measured_point_cap_compliant) if route=='MCQ' else (
                            pd.notna(mean) and pd.notna(memory) and mean<=lat and memory<=r.allocated_VRAM_cap_bytes)
                        upper=route_bounds.loc[sid,'global_latency_upper']
                        if route=='MCQ':
                            memory=route_bounds.loc[sid,'effective_peak_allocated_bytes']
                            pmean=probability.loc[sid,'mean_ms_per_context']
                            point=point and truth(route_bounds.loc[sid,'full_probability_timing_complete']) and pd.notna(pmean) and pmean<=lat and pd.notna(memory) and memory<=r.allocated_VRAM_cap_bytes
                        strict=point and pd.notna(upper) and upper<=lat
                        state=3 if strict else (2 if point else 1)
                    matrix[pi,ci]=state
                    records.append({'route':route,'target_index':target,'policy':policy,'cap_id':cap,
                        'latency_cap_ms':lat,'allocated_VRAM_cap_bytes':r.allocated_VRAM_cap_bytes,
                        'display_state':state,'frozen_development_feasible':feasible})
            ax.imshow(matrix,cmap=ListedColormap(palette),norm=BoundaryNorm(np.arange(-.5,4.5),4),aspect='auto')
            ax.set_xticks(range(len(caps)),caplabels,fontsize=7)
            ax.set_yticks(range(len(policies)),[LABELS.get(p,p) for p in policies],fontsize=8)
            ax.set_title(f'{route}: {NAMES[route,int(target)]} — all {len(caps)} caps',fontsize=11)
            ax.set_xlabel('Cap columns: latency (ms), allocated VRAM (GiB)',fontsize=9)
            ax.set_xticks(np.arange(-.5,len(caps),1),minor=True);ax.set_yticks(np.arange(-.5,len(policies),1),minor=True)
            ax.grid(which='minor',color='white',linewidth=1);ax.tick_params(which='minor',bottom=False,left=False)
    fig.suptitle('Frozen choices: development coverage and actual fresh cap compliance\n'
        'All cells retained; no re-selection or removal of failed cap cells',fontsize=12)
    fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(palette,[
        'Development infeasible','Fresh point cap breach / missing complete timing',
        'Point cap met; conservative latency bound fails','Conservative latency + observed memory cap met'])],
        loc='outside lower center',ncol=2,fontsize=9)
    for ext in ['png','pdf']:fig.savefig(out/f'resource_cap_compliance_grid.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig);pd.DataFrame(records).to_csv(out/'resource_cap_compliance_grid.csv',index=False)
    outputs={p.name:sha(p) for pattern in ['resource_cap_confirmation_gains.*','resource_cap_compliance_grid.*'] for p in out.glob(pattern)}
    (out/'resource_cap_figures_manifest.json').write_text(json.dumps({'inputs':{str(p.relative_to(run)):sha(p) for p in sources},
        'outputs':outputs,'code_sha256':sha(Path(__file__)),
        'scope':'Descriptive plots of frozen routes. MCQ effective latency upper also requires the later probability-materialization audit and can only tighten earlier coordinate bounds. Observed memory is not a future OOM guarantee. Orange contrast intervals are diagnostic, not additional primary claims.'},indent=2)+'\n')
    print(out/'resource_cap_confirmation_gains.png');print(out/'resource_cap_compliance_grid.png')


if __name__=='__main__':main()
