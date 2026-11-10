#!/usr/bin/env python3
"""Six-model discovery and portable fresh-cohort fit/freeze/evaluation runner.

All fits and choices finish and are hash-frozen before evaluation arrays are
constructed. Every policy gets the same per-pair adapter/codec envelope. Global
peer contexts have no target supervision. Old experiments are only read.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, itertools, json, os, resource, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from adapters_v2 import core, decode_values, predict, linear_candidates, FAMILIES
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import run_selection as old
ROOT=Path(__file__).resolve().parents[1]
KIT=ROOT/'experiments/experiment_config'
CONFIG=json.loads((KIT/'configs/campaign.json').read_text())
MODELS=tuple(m['model_id'] for m in CONFIG['model_registry'])
REVISIONS={m['model_id']:m['revision'] for m in CONFIG['model_registry']}
PRIMARY=(1024,2048,8192,16384)
CAPS=(256,512,1024,2048,4096,8192,16384,32768,65536)
SCOPES={'krr':('krr',),'legacy':('convex','nnls','krr'),'strengthened':FAMILIES}
POLICIES=('fixed_native_full','fixed_unbounded_best','fixed_mean_budget','fixed_minimax_validation_regret')
PROTOCOL='replacement_v2_full_family_and_deployment'
sha=old.sha;atomic_json=old.atomic_json;digest_json=old.digest_json
_DATA={};_ARGS=None;_CONTRACT=None

def log(s):print(time.strftime('%Y-%m-%d %H:%M:%S'),s,flush=True)
def pname(p):return '-'.join(map(str,p))
def cell_dir(out,seed,task,target,q):return Path(out)/'cells'/f'seed_{seed}'/task/f'target_{target}'/f'q_{q}'
def context_path(out,seed,task,pair):return Path(out)/'public_contexts'/f'seed_{seed}'/task/f'peers_{pname(pair)}.npz'
def arrays(manifest,responses,dimensions,seeds,tasks):
    out={}
    for seed in seeds:
      for task in tasks:
        splits={}
        for split in ('fit','validation','iid_test'):
          rows=manifest[(manifest.seed==seed)&(manifest.task_id==task)&(manifest.split==split)].sort_values('item_id').reset_index(drop=True)
          if not len(rows):raise ValueError(f'Empty role {seed}/{task}/{split}')
          # Test values deliberately never materialized by fitting workers.
          splits[split]={'rows':rows}
          if split!='iid_test':splits[split]['models']=join_models(rows,responses,dimensions[task])
        out[(seed,task)]=splits
    return out

def join_models(rows,responses,dim):
    result=[]
    for model in MODELS:
        joined=rows[['task_id','item_id','prompt_hash']].merge(responses[responses.model_id==model],on=['task_id','item_id','prompt_hash'],how='left',sort=False,validate='one_to_one',indicator=True)
        if not (joined._merge=='both').all():raise ValueError('Missing responses '+model)
        x=joined[[f'coord_{i}' for i in range(dim)]].to_numpy(float)
        if not np.isfinite(x).all():raise ValueError('Nonfinite active coordinates')
        result.append(x)
    return np.stack(result,axis=1)

def load_inputs(args):
    files=sorted(args.manifest_dir.glob('split_seed_*.parquet'))
    if args.seeds:files=[f for f in files if int(f.stem.rsplit('_',1)[1]) in args.seeds]
    if not files:raise ValueError('No manifests')
    mf=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
    mf=mf[mf.split.isin(('fit','validation','iid_test'))].copy()
    if args.tasks:mf=mf[mf.task_id.isin(args.tasks)]
    if mf.duplicated(['seed','task_id','item_id']).any():raise ValueError('Duplicate manifest keys')
    if (mf.groupby(['seed','task_id','item_cluster']).split.nunique()>1).any():raise ValueError('Cluster role leakage')
    dims=json.loads(args.dimensions_json.read_text()) if args.dimensions_json else {t:v['contrast_dim'] for t,v in CONFIG['tasks'].items()}
    responses,rhash=old.read_responses([args.response_dir],MODELS,REVISIONS)
    required_mf=mf[mf.split!='iid_test'] if args.test_response_dir else mf
    missing=old.coverage(required_mf,responses,MODELS)
    if len(missing):raise ValueError(f'Missing {len(missing)} response keys')
    # Every model must cover exactly the same task/item/prompt keys.
    keys=responses[['task_id','item_id','prompt_hash']].drop_duplicates()
    if len(responses)!=len(keys)*len(MODELS):raise ValueError('Response banks are not a complete six-model Cartesian product')
    seeds=sorted(int(s) for s in mf.seed.unique());tasks=sorted(mf.task_id.unique())
    codepaths=[Path(__file__),Path(__file__).with_name('adapters_v2.py'),ROOT/'benchmarks/selection_core.py',ROOT/'benchmarks/run_selection.py']
    prov={str(p):sha(p) for p in sorted(args.response_dir.glob('*.provenance.json'))}
    protocol={'protocol_id':PROTOCOL,'data_role':args.data_role,'registry':CONFIG['model_registry'],'dimensions':dims,'targets':args.targets,'q':args.q,'seeds':seeds,'tasks':tasks,'primary_caps':PRIMARY,'sensitivity_caps':CAPS,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in codepaths},'manifest_hashes':{str(f):sha(f) for f in files},'response_hashes':rhash,'provenance_hashes':prov,'families':FAMILIES,'scopes':SCOPES,'lambda_grid':core.LAMBDAS,'rank_grid':core.RANKS,'codecs':core.CODECS,'fixed_comparators':POLICIES,'no_test_before_all_freezes':True,'separate_sealed_test_bank':bool(args.test_response_dir),'tie_rule':'validation TV, actual bits, family, scientific rank, descending lambda, codec, state digest; global pair IDs final','conditional_ci':'paired cluster bootstrap; within-cell 20-contrast Bonferroni intervals for four caps and five comparators; exploratory across cells/seeds','python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}
    return mf,responses,dims,seeds,tasks,protocol

def key(r):return (r['validation_tv'],r['actual_bits'],r['family'],r['rank'],-core.LAMBDAS[r['lambda_index']],r['codec'],r['state_digest'])

def enumerate_pair(fit,val,target,pair,context,state_dir):
    y=fit[:,target,:];vy=val[:,target,:];vprob=core.probabilities(vy)
    projected=context.eigenvectors.T@y
    ranks=sorted(set([r for r in core.RANKS if r<len(fit)]+[len(fit)]))
    ledger=[];best={};blobs={}
    def consider(matrix,family,codec,li,rank,native=False):
        try:
            transmitted=decode_values(matrix,family,codec)
            encoded=core.encode(matrix,family,pair,codec,li)
            # Independent decoder is exercised for every candidate, including
            # signed low-bit values and the NNLS non-simplex distinction.
            decoded,actual=core.decode(encoded.blob)
            if not np.array_equal(actual,transmitted):raise AssertionError('Codec fast path differs from literal decoding')
        except ValueError as exc:
            if str(exc) not in ('Native precision overflow','Scale overflow'):raise
            ledger.append({'family':family,'pair':list(pair),'codec':codec,'rank':rank,'lambda_index':li,'status':'unrepresentable','error':str(exc)})
            return
        pred=predict(decoded,actual,val,context,context.validation_projection)
        err=float(np.abs(vprob-core.probabilities(pred)).sum(1).mean()/2)
        if not np.isfinite(err):raise ValueError('Nonfinite validation loss')
        rec={'family':family,'pair':list(pair),'q':len(pair),'codec':codec,'rank':rank,'payload_rows':len(matrix),'lambda_index':li,'actual_bits':encoded.bits,'meaningful_bits':encoded.meaningful_bits,'padding_bits':encoded.bits-encoded.meaningful_bits,'validation_tv':err,'state_digest':encoded.digest,'native_full_precision':bool(native),'status':'eligible'}
        ledger.append(rec)
        budgets=[b for b in CAPS if encoded.bits<=b]+[-2]+([-1] if native else [])
        for b in budgets:
            k=(family,b)
            if k not in best or key(rec)<key(best[k]):
                best[k]=rec;blobs[encoded.digest]=encoded.blob
    for fam in ('convex','nnls'):
        weights=core.fit_weights(fit,y,pair,fam).reshape(-1,1)
        for codec in core.CODECS:consider(weights,fam,codec,0,len(pair),codec=='native_fp64')
    for li,lam in enumerate(core.LAMBDAS):
        spectral=projected/(context.eigenvalues[:,None]+len(fit)*lam)
        for rank in ranks:
            for codec in core.CODECS:consider(spectral[:rank],'krr',codec,li,rank,rank==len(fit) and codec=='native_fp64')
    max_residual_rank=min(8,len(pair)*y.shape[1],y.shape[1])
    for family,matrix,li,rank in linear_candidates(fit,y,pair,context):
        for codec in core.CODECS:
            native=codec=='native_fp64' and (family=='linear_ridge' or rank==max_residual_rank)
            consider(matrix,family,codec,li,rank,native)
    selected=[r|{'budget_bits':int(b),'selection_kind':'pair_family'} for (f,b),r in sorted(best.items())]
    selected_digests={r['state_digest'] for r in selected}
    state_dir.mkdir(parents=True,exist_ok=True)
    for d in selected_digests:(state_dir/f'{d}.bin').write_bytes(blobs[d])
    return selected,ledger

def fit_job(job):
    seed,task,pair=job;start=time.time();out=_ARGS.output
    data=_DATA[(seed,task)];fit=data['fit']['models'];val=data['validation']['models']
    path=context_path(out,seed,task,pair);path.parent.mkdir(parents=True,exist_ok=True)
    # Context identity only uses the selected peers, fit/validation keys and
    # peer revisions. Target identity/response arrays are intentionally absent.
    fp={'pair':pair,'revisions':[REVISIONS[MODELS[i]] for i in pair],
        'fit_keys':data['fit']['rows'][['item_id','prompt_hash']].to_dict('records'),
        'val_keys':data['validation']['rows'][['item_id','prompt_hash']].to_dict('records'),
        'fit_peer_bytes':hashlib.sha256(fit[:,pair,:].tobytes()).hexdigest(),
        'val_peer_bytes':hashlib.sha256(val[:,pair,:].tobytes()).hexdigest(),
        'context_code':sha(ROOT/'benchmarks/selection_core.py')}
    fp_hash=digest_json(fp);meta=path.with_suffix('.json')
    if path.exists():
        m=json.loads(meta.read_text())
        if m['fingerprint']!=fp_hash or m['sha256']!=sha(path):raise ValueError('Public context drift')
        with np.load(path) as c:context=core.Context(pair,c['mean'],c['scale'],float(c['sigma']),c['features'],c['eigenvalues'],c['eigenvectors'],c['validation_projection'])
    else:
        context=core.make_context(fit,val,pair)
        temp=path.with_suffix('.npz.tmp')
        with temp.open('wb') as f:np.savez(f,mean=context.mean,scale=context.scale,sigma=context.sigma,features=context.features,eigenvalues=context.eigenvalues,eigenvectors=context.eigenvectors,validation_projection=context.validation_projection)
        temp.replace(path)
        atomic_json(meta,{'fingerprint':fp_hash,'fingerprint_inputs':fp,'sha256':sha(path),'contains_target_outputs':False,'deployment_arrays':['mean','scale','sigma','features','eigenvectors'],'experiment_only_arrays':['eigenvalues','validation_projection'],'disk_bytes':path.stat().st_size,'deployment_array_bytes':sum(x.nbytes for x in (context.mean,context.scale,context.features,context.eigenvectors))+8})
    completed=[]
    for target in _ARGS.targets:
        if target in pair:continue
        d=cell_dir(out,seed,task,target,len(pair))/'pairs'/pname(pair);done=d/'fit_complete.json'
        if done.exists():
            record=json.loads(done.read_text())
            if record['contract']!=_CONTRACT:raise ValueError('Fit code/input contract drift')
            for rel,h in record['artifacts'].items():
                if sha(d/rel)!=h:raise ValueError('Fit artifact changed')
            completed.append(target);continue
        d.mkdir(parents=True,exist_ok=True)
        rows,ledger=enumerate_pair(fit,val,target,pair,context,d/'states')
        pd.DataFrame(ledger).to_parquet(d/'validation_candidates.parquet',index=False)
        atomic_json(d/'pair_selections.json',rows)
        artifacts={str(p.relative_to(d)):sha(p) for p in [d/'pair_selections.json',d/'validation_candidates.parquet']+sorted((d/'states').glob('*.bin'))}
        atomic_json(done,{'contract':_CONTRACT,'target':target,'pair':pair,'context':str(path.relative_to(out)),'context_sha256':sha(path),'artifacts':artifacts,'candidates':len(ledger),'created_unix':time.time()})
        completed.append(target)
        log(f'fit seed={seed} task={task} peers={pair} target={target}; candidates={len(ledger)} elapsed={time.time()-start:.1f}s')
    return {'seed':seed,'task':task,'pair':pair,'targets':completed,'elapsed_seconds':time.time()-start,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'context_bytes':path.stat().st_size}

def select_policies(rows,target,q):
    """No test fields accepted; all policies see exactly the same menu."""
    if any(any('test' in k or 'oracle' in k for k in r) for r in rows):raise ValueError('Test data in selection API')
    pairs=list(itertools.combinations([i for i in range(6) if i!=target],q))
    selected=[]
    for scope,families in SCOPES.items():
        env={}
        for r in rows:
            if r['family'] not in families:continue
            k=(tuple(r['pair']),r['budget_bits'])
            if k not in env or key(r)<key(env[k]):env[k]=r
        # KRR alone may be infeasible at the smallest sensitivity caps; record
        # those cells explicitly rather than silently comparing only some pairs.
        feasible=[b for b in CAPS if all((p,b) in env for p in pairs)]
        if not all(b in feasible for b in PRIMARY):raise ValueError('A primary cap is infeasible for a whole menu')
        dynamic={b:min((env[p,b] for p in pairs),key=lambda r:(key(r),tuple(r['pair']))) for b in feasible}
        native=min(pairs,key=lambda p:(key(env[p,-1]),p));unbounded=min(pairs,key=lambda p:(key(env[p,-2]),p))
        mean=min(pairs,key=lambda p:(sum(env[p,b]['validation_tv'] for b in PRIMARY)/len(PRIMARY),p))
        minimax=min(pairs,key=lambda p:(max(env[p,b]['validation_tv']-dynamic[b]['validation_tv'] for b in PRIMARY),sum(env[p,b]['validation_tv'] for b in PRIMARY),p))
        fixed=dict(zip(POLICIES,(native,unbounded,mean,minimax)))
        for b in CAPS:
            if b not in feasible:
                selected.append({'scope':scope,'budget_bits':b,'status':'infeasible_complete_family','selection_kind':'policy','policy':'budget_aware','q':q})
                continue
            for policy,row in [('budget_aware',dynamic[b])]+[(name,env[p,b]) for name,p in fixed.items()]:
                selected.append(row|{'scope':scope,'budget_bits':b,'selection_kind':'policy','policy':policy,'selected_before_test':True})
            nnls=[r for r in rows if r['family']=='nnls' and r['budget_bits']==b]
            if nnls:selected.append(min(nnls,key=lambda r:(key(r),tuple(r['pair'])))|{'scope':scope,'budget_bits':b,'selection_kind':'policy','policy':'budget_aware_nnls','selected_before_test':True})
    return selected

def freeze_cell(out,seed,task,target,q,contract):
    d=cell_dir(out,seed,task,target,q);f=d/'selection_freeze.json'
    if f.exists():
        freeze=json.loads(f.read_text())
        if freeze['contract']!=contract:raise ValueError('Freeze code/input drift')
        return f
    rows=[];artifacts={};contexts={}
    for pair in itertools.combinations([i for i in range(6) if i!=target],q):
        pdirectory=d/'pairs'/pname(pair);complete=json.loads((pdirectory/'fit_complete.json').read_text())
        if complete['contract']!=contract:raise ValueError('Pair contract drift')
        for rel,h in complete['artifacts'].items():
            p=pdirectory/rel
            if sha(p)!=h:raise ValueError('Pair artifact drift')
            artifacts[str(p.relative_to(d))]=h
        rows.extend(json.loads((pdirectory/'pair_selections.json').read_text()))
        cp=context_path(out,seed,task,pair);contexts[pname(pair)]={'path':os.path.relpath(cp,d),'sha256':sha(cp)}
    decisions=select_policies(rows,target,q)
    atomic_json(f,{'protocol_id':PROTOCOL,'contract':contract,'seed':seed,'task':task,'target':target,'target_id':MODELS[target],'target_revision':REVISIONS[MODELS[target]],'q':q,'selections':rows+decisions,'artifacts':artifacts,'contexts':contexts,'test_evaluated':False,'created_unix':time.time()})
    return f

def validate_freeze(f,contract):
    freeze=json.loads(f.read_text())
    if freeze['contract']!=contract:raise ValueError('Freeze contract drift')
    for rel,h in freeze['artifacts'].items():
        if sha(f.parent/rel)!=h:raise ValueError('Frozen state/ledger changed')
    for c in freeze['contexts'].values():
        if sha((f.parent/c['path']).resolve())!=c['sha256']:raise ValueError('Frozen public context changed')
    return freeze

def paired_intervals(differences,clusters,draws,seed):
    unique,inv=np.unique(np.asarray(clusters).astype(str),return_inverse=True)
    count=np.bincount(inv);sums=np.zeros((len(unique),differences.shape[1]))
    np.add.at(sums,inv,differences)
    rng=np.random.default_rng(seed);est=np.empty((draws,differences.shape[1]))
    for start in range(0,draws,128):
        ix=rng.integers(len(unique),size=(min(128,draws-start),len(unique)))
        est[start:start+len(ix)]=sums[ix].sum(1)/count[ix].sum(1)[:,None]
    return np.quantile(est,[.025,.975,.05/(2*20),1-.05/(2*20)],axis=0)

def evaluate_cell(job):
    seed,task,target,q=job;out=_ARGS.output;d=cell_dir(out,seed,task,target,q);f=d/'selection_freeze.json'
    freeze=validate_freeze(f,_CONTRACT);done=d/'evaluation_complete.json'
    if done.exists():
        v=json.loads(done.read_text())
        if v['freeze_sha256']!=sha(f) or v['bootstrap_draws']!=_ARGS.bootstrap_draws:raise ValueError('Evaluation settings drift')
        for rel,h in v['outputs'].items():
            if sha(d/rel)!=h:raise ValueError('Evaluation output changed')
        return str(done)
    data=_DATA[(seed,task)]['iid_test'];peers=data['models'];rows=data['rows'];target_array=peers[:,target,:]
    loss={};predictions=[];contexts={};projections={}
    selections=[r for r in freeze['selections'] if r['status']=='eligible']
    for row in selections:
        digest=row['state_digest']
        if digest in loss:continue
        pair=tuple(row['pair']);state=d/'pairs'/pname(pair)/'states'/f'{digest}.bin'
        encoded,matrix=core.decode(state.read_bytes())
        if encoded.digest!=digest or encoded.bits!=row['actual_bits'] or encoded.pair!=pair:raise ValueError('State metadata differs')
        if pair not in contexts:
            cp=(d/freeze['contexts'][pname(pair)]['path']).resolve()
            with np.load(cp) as c:contexts[pair]=core.Context(pair,c['mean'],c['scale'],float(c['sigma']),c['features'],c['eigenvalues'],c['eigenvectors'],np.empty((0,0)))
        if encoded.family=='krr' and pair not in projections:projections[pair]=contexts[pair].project(peers)
        pred=predict(encoded,matrix,peers,contexts[pair],projections.get(pair));loss[digest]=core.tv(target_array,pred)
        frame=rows[[c for c in ('item_id','item_cluster','domain','gold','prompt_hash') if c in rows]].copy()
        frame['state_digest']=digest;frame['tv_loss']=loss[digest]
        pp=core.probabilities(pred);tp=core.probabilities(target_array)
        frame['kl_loss']=(tp*(np.log(np.maximum(tp,1e-300))-np.log(np.maximum(pp,1e-300)))).sum(1)
        frame['argmax_disagreement']=(pp.argmax(1)!=tp.argmax(1)).astype(np.uint8)
        for j in range(pred.shape[1]):frame[f'prediction_coord_{j}']=pred[:,j]
        predictions.append(frame)
    meta={'seed':seed,'task_id':task,'target_id':MODELS[target],'target_global_id':target,'target_revision':REVISIONS[MODELS[target]],'protocol_id':PROTOCOL,'data_role':_ARGS.data_role,'test_cluster_count':int(rows.item_cluster.nunique())}
    scores=[r|meta|{'test_tv':float(loss[r['state_digest']].mean()),'peer_ids':[MODELS[i] for i in r['pair']],'peer_revisions':[REVISIONS[MODELS[i]] for i in r['pair']],'public_context_digest':freeze['contexts'][pname(r['pair'])]['sha256'],'selected_before_test':True} for r in selections]
    pd.concat(predictions,ignore_index=True).to_parquet(d/'item_predictions_losses.parquet',index=False)
    pd.DataFrame(scores).to_parquet(d/'test_scores.parquet',index=False)
    pd.DataFrame(scores).to_csv(d/'test_scores.csv',index=False)
    gains=[]
    for scope in SCOPES:
        differences=[];records=[]
        for b in CAPS:
            selected={r['policy']:r for r in scores if r['selection_kind']=='policy' and r['scope']==scope and r['budget_bits']==b}
            if not selected:continue
            joint=selected['budget_aware']
            for comp in POLICIES+('budget_aware_nnls',):
                fixed=selected[comp];difference=loss[fixed['state_digest']]-loss[joint['state_digest']]
                differences.append(difference)
                records.append(meta|{'q':q,'scope':scope,'budget_bits':b,'comparator':comp,'fixed_pair':fixed['pair'],'joint_pair':joint['pair'],'fixed_tv':fixed['test_tv'],'joint_tv':joint['test_tv'],'gain_tv':float(difference.mean()),'relative_gain':float(difference.mean()/fixed['test_tv']) if fixed['test_tv'] else 0.,'fixed_state_digest':fixed['state_digest'],'joint_state_digest':joint['state_digest'],'fixed_actual_bits':fixed['actual_bits'],'joint_actual_bits':joint['actual_bits'],'fixed_family':fixed['family'],'joint_family':joint['family'],'validation_gain':fixed['validation_tv']-joint['validation_tv'],'uncertainty':'paired cluster bootstrap conditional on split and frozen choices; simultaneous within 4 primary caps x 5 comparators; sensitivity intervals descriptive'})
        if records:
            ci=paired_intervals(np.stack(differences,axis=1),rows.item_cluster.to_numpy(),_ARGS.bootstrap_draws,seed)
            for i,r in enumerate(records):gains.append(r|{'ci_low':float(ci[0,i]),'ci_high':float(ci[1,i]),'simultaneous_ci_low':float(ci[2,i]),'simultaneous_ci_high':float(ci[3,i]),'paired_cluster_sd':float(np.std(differences[i],ddof=1))})
    pd.DataFrame(gains).to_csv(d/'policy_gains.csv',index=False)
    outputs=['item_predictions_losses.parquet','test_scores.parquet','test_scores.csv','policy_gains.csv']
    atomic_json(done,{'freeze_sha256':sha(f),'bootstrap_draws':_ARGS.bootstrap_draws,'outputs':{p:sha(d/p) for p in outputs},'created_unix':time.time()})
    log(f'evaluated {seed}/{task}/target={target}/q={q}')
    return str(done)

def aggregate(out,cells):
    score=pd.concat([pd.read_parquet(cell_dir(out,*cell)/'test_scores.parquet') for cell in cells],ignore_index=True)
    gains=pd.concat([pd.read_csv(cell_dir(out,*cell)/'policy_gains.csv') for cell in cells],ignore_index=True)
    score.to_parquet(out/'all_scores.parquet',index=False)
    score[score.selection_kind=='pair_family'].to_csv(out/'all_pairs_by_budget.csv',index=False)
    gains.to_csv(out/'policy_gains.csv',index=False)
    cols=['task_id','target_id','q','scope','budget_bits','comparator']
    summary=gains.groupby(cols,as_index=False).agg(split_count=('gain_tv','size'),mean_gain_tv=('gain_tv','mean'),min_gain_tv=('gain_tv','min'),max_gain_tv=('gain_tv','max'),positive_splits=('gain_tv',lambda x:int((x>1e-15).sum())),zero_splits=('gain_tv',lambda x:int((abs(x)<=1e-15).sum())),negative_splits=('gain_tv',lambda x:int((x< -1e-15).sum())),mean_validation_gain=('validation_gain','mean'),mean_relative_gain=('relative_gain','mean'))
    summary['uncertainty_note']='Overlapping seeds; summary is descriptive, not independent replication'
    summary.to_csv(out/'all_cells_summary.csv',index=False)
    decisions=[]
    for cell in cells:
        freeze=json.loads((cell_dir(out,*cell)/'selection_freeze.json').read_text())
        decisions.extend({k:freeze[k] for k in ('seed','task','target','q')}|r for r in freeze['selections'] if r['selection_kind']=='policy')
    with (out/'validation_policy_decisions.jsonl').open('w') as f:
        for r in decisions:f.write(json.dumps(r,sort_keys=True)+'\n')
    # A test oracle only over adapters already frozen using development data.
    head=[]
    for (seed,task,target,q),s in score.groupby(['seed','task_id','target_id','q']):
      for scope,families in SCOPES.items():
       for b in CAPS:
        candidates=s[(s.selection_kind=='pair_family')&(s.budget_bits==b)&s.family.isin(families)]
        if candidates.empty:continue
        perpair=[]
        for _,group in candidates.groupby(candidates.pair.apply(lambda x:tuple(x))):
            r=min(group.to_dict('records'),key=key);perpair.append(r)
        oracle=min(perpair,key=lambda r:r['test_tv'])
        joint=s[(s.selection_kind=='policy')&(s.scope==scope)&(s.policy=='budget_aware')&(s.budget_bits==b)]
        if joint.empty:continue
        head.append({'seed':seed,'task_id':task,'target_id':target,'q':q,'scope':scope,'budget_bits':b,'diagnostic_only':True,'joint_test_tv':float(joint.test_tv.iloc[0]),'retrospective_best_frozen_pair_tv':oracle['test_tv'],'remaining_test_oracle_headroom':float(joint.test_tv.iloc[0]-oracle['test_tv']),'oracle_pair':list(oracle['pair'])})
    pd.DataFrame(head).to_csv(out/'headroom_diagnostic.csv',index=False)
    # Registered follow-up ranking: mean validation gain vs fixed mean across
    # primary budgets and seeds, distinct target IDs, primary q2 stronger menu.
    rank=gains[(gains.q==2)&(gains.scope=='strengthened')&(gains.comparator=='fixed_mean_budget')&gains.budget_bits.isin(PRIMARY)].groupby(['target_id','task_id'],as_index=False).validation_gain.mean().sort_values(['validation_gain','target_id','task_id'],ascending=[False,True,True])
    rank['selected_for_confirmation']=False;targets=set()
    for ix,r in rank.iterrows():
        if r.target_id not in targets and len(targets)<2:rank.loc[ix,'selected_for_confirmation']=True;targets.add(r.target_id)
    rank.to_csv(out/'followup_validation_ranking.csv',index=False)
    return len(score),len(gains)

def main():
    global _ARGS,_DATA,_CONTRACT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest-dir',type=Path,default=ROOT/'benchmarks/splits');p.add_argument('--response-dir',type=Path,default=ROOT/'benchmarks/response_cache/recompute_all');p.add_argument('--dimensions-json',type=Path)
    p.add_argument('--test-response-dir',type=Path,help='Separate sealed test bank: not read or required until global selection freeze')
    p.add_argument('--output',type=Path,default=ROOT/'experiments/runs/v2/phase_a');p.add_argument('--stage',choices=['plan','fit','evaluate','run'],default='plan');p.add_argument('--seeds',type=int,nargs='+');p.add_argument('--tasks',nargs='+');p.add_argument('--targets',type=int,nargs='+',default=list(range(6)));p.add_argument('--q',type=int,nargs='+',default=[2]);p.add_argument('--workers',type=int,default=4);p.add_argument('--blas-threads',type=int,default=4);p.add_argument('--bootstrap-draws',type=int,default=2000);p.add_argument('--data-role',default='exploratory_existing_cache');p.add_argument('--max-context-jobs',type=int)
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    if not set(args.targets)<=set(range(6)) or not set(args.q)<={1,2,3}:p.error('Invalid target/q')
    mf,responses,dims,seeds,tasks,protocol=load_inputs(args);contract=digest_json(protocol)
    planpath=args.output/'run_plan.json'
    if planpath.exists() and json.loads(planpath.read_text())['contract']!=contract:raise ValueError('Run contract changed; choose a new output')
    jobs=[(s,t,pair) for s in seeds for t in tasks for q in args.q for pair in itertools.combinations(range(6),q) if any(target not in pair for target in args.targets)]
    cells=[(s,t,target,q) for s in seeds for t in tasks for target in args.targets for q in args.q]
    plan={'contract':contract,'protocol':protocol,'context_jobs':len(jobs),'cells':len(cells),'target_pair_fits':sum(len(list(itertools.combinations([i for i in range(6) if i!=target],q))) for s,t,target,q in cells),'workers':args.workers,'blas_threads':args.blas_threads,'bootstrap_draws':args.bootstrap_draws,'created_unix':time.time()}
    atomic_json(planpath,plan);log(f'plan cells={len(cells)} contexts={len(jobs)} fits={plan["target_pair_fits"]}')
    if args.stage=='plan':return
    _ARGS=args;_CONTRACT=contract;_DATA=arrays(mf,responses,dims,seeds,tasks)
    with threadpool_limits(limits=args.blas_threads):
      if args.stage in ('fit','run'):
        running=jobs[:args.max_context_jobs] if args.max_context_jobs else jobs
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
          futures={pool.submit(fit_job,j):j for j in running}
          profiles=[]
          for future in concurrent.futures.as_completed(futures):
            job=futures[future]
            try:profiles.append(future.result())
            except Exception as exc:
                with (args.output/'failures.jsonl').open('a') as f:f.write(json.dumps({'stage':'fit','job':job,'error':repr(exc),'time':time.time()})+'\n')
                raise
            atomic_json(args.output/'fit_progress.json',{'completed_context_jobs':len(profiles),'launched_context_jobs':len(running),'planned_context_jobs':len(jobs),'profiles':profiles,'updated_unix':time.time()})
        if args.max_context_jobs:return
        paths=[freeze_cell(args.output,*cell,contract) for cell in cells]
        atomic_json(args.output/'all_selections_frozen.json',{'contract':contract,'cells':len(paths),'freeze_hashes':{str(f.relative_to(args.output)):sha(f) for f in paths},'created_unix':time.time()})
        log('ALL selections frozen before any evaluation arrays')
      if args.stage in ('evaluate','run'):
        globalfreeze=json.loads((args.output/'all_selections_frozen.json').read_text())
        if globalfreeze['contract']!=contract or globalfreeze['cells']!=len(cells):raise ValueError('Global freeze incomplete')
        for rel,h in globalfreeze['freeze_hashes'].items():
            if sha(args.output/rel)!=h:raise ValueError('Global freeze drift')
            validate_freeze(args.output/rel,contract)
        test_responses=responses
        if args.test_response_dir:
            test_responses,test_hashes=old.read_responses([args.test_response_dir],MODELS,REVISIONS)
            missing_test=old.coverage(mf[mf.split=='iid_test'],test_responses,MODELS)
            if len(missing_test):raise ValueError(f'Missing sealed test keys: {len(missing_test)}')
            input_path=args.output/'evaluation_inputs.json'
            evidence={'response_hashes':test_hashes,'global_freeze_sha256':sha(args.output/'all_selections_frozen.json')}
            if input_path.exists() and json.loads(input_path.read_text())!=evidence:raise ValueError('Sealed test bank changed')
            atomic_json(input_path,evidence)
        for (seed,task),data in _DATA.items():data['iid_test']['models']=join_models(data['iid_test']['rows'],test_responses,dims[task])
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
          done=list(pool.map(evaluate_cell,cells))
        nscore,ngain=aggregate(args.output,cells)
        atomic_json(args.output/'complete.json',{'contract':contract,'cells':len(cells),'score_rows':nscore,'gain_rows':ngain,'finished_unix':time.time(),'scientific_outcome':'see all_cells_summary; completion is not a positive claim'})
        log(f'COMPLETE cells={len(cells)} score_rows={nscore} gains={ngain}')

if __name__=='__main__':main()
