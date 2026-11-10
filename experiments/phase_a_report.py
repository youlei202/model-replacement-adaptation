#!/usr/bin/env python3
"""Measured complete Phase A tables and standard scientific figures."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PRIMARY=[1024,2048,8192,16384]
LABELS={'budget_aware':'Joint','fixed_native_full':'Fixed native full','fixed_unbounded_best':'Fixed unbounded best','fixed_mean_budget':'Fixed mean budget','fixed_minimax_validation_regret':'Fixed minimax regret','budget_aware_nnls':'Adaptive NNLS'}
COLORS={'budget_aware':'black','fixed_native_full':'#8C6D31','fixed_unbounded_best':'#756BB1','fixed_mean_budget':'#D95F0E','fixed_minimax_validation_regret':'#238B45','budget_aware_nnls':'#2171B5'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'runs/v2/phase_a');a=p.parse_args();out=a.output
 complete=json.loads((out/'complete.json').read_text());plan=json.loads((out/'run_plan.json').read_text());g=pd.read_csv(out/'policy_gains.csv');summary=pd.read_csv(out/'all_cells_summary.csv');scores=pd.read_parquet(out/'all_scores.parquet')
 models=[x['model_id'] for x in plan['protocol']['registry']];tasks=plan['protocol']['tasks'];names=[m.split('/')[-1] for m in models]
 primary=g[(g.q==2)&(g.scope=='strengthened')&g.budget_bits.isin(PRIMARY)&(g.comparator!='budget_aware_nnls')]
 s=summary[(summary.q==2)&(summary.scope=='strengthened')&summary.budget_bits.isin(PRIMARY)&(summary.comparator!='budget_aware_nnls')]
 lines=['# Complete Phase A results','',f"Executed {complete['cells']} target/task/seed/q cells: 90 primary q=2 cells (900 pair fits), plus q=1 and q=3 controls, total 2250 interface fits. Six targets, all eligible interfaces, five adapters, all inherited lambda/rank/precision choices, nine caps and unrestricted candidates were evaluated. All decisions were frozen before test evaluation.",'','These are exploratory results on previously inspected queries. Five split seeds overlap; mean/min/max and sign counts across them are descriptive. The ordinary paired bootstrap is conditional on each fitted/selected system. Simultaneous intervals cover four primary caps × five comparators within a given cell and scope, not a familywide claim across all cells, q values, and scopes.','', 'Gain = test TV(fixed) − test TV(joint); positive favors joint. Every fixed interface retunes its adapter at each cap. Headline menu: convex, NNLS, KRR, linear ridge with bias, and NNLS plus low-rank residual. The residual menu includes zero correction through the NNLS candidate.','', '| Target | Task | Cap (bits) | Fixed comparator | Mean gain | Seed min | Seed max | + / 0 / − |','|---|---|---:|---|---:|---:|---:|---|']
 for r in s.sort_values(['target_id','task_id','budget_bits','comparator']).itertuples():
  lines.append(f'| {r.target_id} | {r.task_id} | {r.budget_bits} | {r.comparator} | {r.mean_gain_tv:.8f} | {r.min_gain_tv:.8f} | {r.max_gain_tv:.8f} | {r.positive_splits}/{r.zero_splits}/{r.negative_splits} |')
 lines += ['', 'The complete tables retain KRR-only, legacy three-family, strengthened five-family menus; all q=1/2/3 controls and all sensitivity caps. `all_pairs_by_budget.csv` contains per-family interfaces; `all_scores.parquet` contains frozen policy choices; `policy_gains.csv` contains all paired contrasts. `predetermined_public_controls.csv` lists all public mean/direct interfaces without selecting a free target-specific winner.','', 'The original-context legacy replay reproduces all 45 original cells / 1490 unique states and public means to 1.11e-16 validation TV. Fresh eigendecompositions rotate nearly null RewardBench eigenspaces: 39 of 2700 legacy candidate comparisons differ above 1e-8, maximum 0.001013 TV. These refits have distinct context hashes and are not substituted into old results. See `refit_numerical_sensitivity.json` and both regression tables.','', 'Binary caps include all encoded adapter bytes and selector/header/padding, but are not total deployment memory caps. Public KRR support/eigenbasis arrays, peer weights, runtime, routing metadata, and resident state require separate measured accounting.']
 (out/'ALL_PHASE_A_RESULTS.md').write_text('\n'.join(lines)+'\n')
 fig,axes=plt.subplots(6,3,figsize=(14,19),sharex=True,constrained_layout=True)
 for ti,model in enumerate(models):
  for tj,task in enumerate(tasks):
   ax=axes[ti,tj];cell=scores[(scores.target_id==model)&(scores.task_id==task)&(scores.q==2)&scores.budget_bits.isin(PRIMARY)]
   pair=cell[cell.selection_kind=='pair_family'].copy();pair['pair_key']=pair.pair.map(lambda p:'-'.join(map(str,p)))
   # Select the family within each pair/cap using validation alone.
   pair=pair.sort_values(['validation_tv','actual_bits','family','rank','lambda_index','codec','state_digest'],ascending=[True,True,True,True,False,True,True]).drop_duplicates(['seed','pair_key','budget_bits'])
   for _,d in pair.groupby('pair_key'):
    y=d.groupby('budget_bits').test_tv.mean().reindex(PRIMARY);ax.plot(PRIMARY,y,color='.8',lw=.8,zorder=1)
   policy=cell[(cell.selection_kind=='policy')&(cell.scope=='strengthened')]
   for name in LABELS:
    d=policy[policy.policy==name].groupby('budget_bits').test_tv.mean().reindex(PRIMARY)
    ax.plot(PRIMARY,d,color=COLORS[name],marker='o',ms=2.5,lw=1.2,label=LABELS[name])
   ax.set_xscale('log',base=2);ax.grid(alpha=.2);ax.set_title(f'{names[ti]}\n{task}',fontsize=9)
   if tj==0:ax.set_ylabel('Mean held-out TV')
   if ti==5:ax.set_xlabel('Encoded-state cap (bits)')
 axes[0,0].legend(fontsize=6,loc='best')
 fig.suptitle('All six targets × three tasks: validation-selected policies and all ten peer-pair envelopes\nMean over five overlapping discovery splits; gray pairs select adapters by validation',fontsize=12)
 for ext in ('png','pdf'):fig.savefig(out/f'error_budget_curves.{ext}',dpi=170)
 plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
 comps=[x for x in LABELS if x.startswith('fixed_')]
 for ax,comp in zip(axes.flat,comps):
  x=primary[primary.comparator==comp].gain_tv.to_numpy();ax.hist(x,bins=31,color=COLORS[comp],edgecolor='white');ax.axvline(0,color='black',lw=1);ax.set_title(f'{LABELS[comp]}\npositive={sum(x>1e-15)}, zero={sum(abs(x)<=1e-15)}, negative={sum(x< -1e-15)}');ax.set_xlabel('TV(fixed) − TV(joint)');ax.set_ylabel('Discovery cell/cap count')
 fig.suptitle('All 18 settings × five overlapping seeds × four primary budgets, q=2 strengthened menu')
 for ext in ('png','pdf'):fig.savefig(out/f'all_cell_gain_distribution.{ext}',dpi=170)
 plt.close(fig)
 evidence={'completed_cells':complete['cells'],'primary_comparisons':len(primary),'max_primary_single_split_gain':float(primary.gain_tv.max()),'min_primary_single_split_gain':float(primary.gain_tv.min()),'primary_mean_gain_by_comparator':primary.groupby('comparator').gain_tv.mean().to_dict(),'largest_cell_mean_gain_by_comparator':s.groupby('comparator').mean_gain_tv.max().to_dict(),'nonzero_comparisons':int((abs(primary.gain_tv)>1e-15).sum()),'strictly_positive_simultaneous_lower_bounds':int((primary.simultaneous_ci_low>0).sum()),'exploratory':True,'fresh_confirmation':False}
 (out/'phase_a_evidence_summary.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
