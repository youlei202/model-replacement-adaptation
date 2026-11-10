import sys,itertools
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import adapters_v2 as a
import phase_a_run as r

@pytest.mark.parametrize('family',a.FAMILIES)
@pytest.mark.parametrize('codec',a.core.CODECS)
def test_literal_decode_fast_parity(family,codec):
    rng=np.random.default_rng(8)
    matrix=rng.normal(size=(17,3)) if family in ('krr','linear_ridge') else rng.normal(size=(15,1))
    if family in ('nnls','convex'):matrix=np.array([[.23],[1.42],[.18]])
    blob=a.core.encode(matrix,family,(0,3,5),codec,6)
    state,actual=a.core.decode(blob.blob)
    assert np.array_equal(actual,a.decode_values(matrix,family,codec))
    assert state.pair==(0,3,5) and state.bits==len(blob.blob)*8
    if family=='nnls':assert not np.isclose(actual.sum(),1)

def test_context_excludes_target():
    rng=np.random.default_rng(3);x=rng.normal(size=(40,6,3));v=rng.normal(size=(12,6,3))
    c=a.core.make_context(x,v,(1,5));x[:,3,:]+=100;v[:,3,:]-=200
    c2=a.core.make_context(x,v,(1,5))
    for name in ('mean','scale','features','eigenvectors','eigenvalues','validation_projection'):
        assert np.array_equal(getattr(c,name),getattr(c2,name))

def test_linear_residual_saved_replay():
    rng=np.random.default_rng(4);x=rng.normal(size=(80,6,3));v=rng.normal(size=(20,6,3));pair=(0,4)
    y=x[:,0,:]*1.5+x[:,4,:]*.4+np.array([2,-1,.5])
    context=a.core.make_context(x,v,pair)
    for family,matrix,li,rank in a.linear_candidates(x,y,pair,context,lambdas=(1e-8,)):
        state=a.core.encode(matrix,family,pair,'native_fp64',li);encoded,m=a.core.decode(state.blob)
        pred=a.predict(encoded,m,v,context)
        assert pred.shape==(20,3) and np.isfinite(pred).all()
        assert np.array_equal(pred,a.predict(encoded,matrix,v,context))
        if family=='linear_ridge' or rank==3:
            assert np.max(np.abs(pred-(v[:,0,:]*1.5+v[:,4,:]*.4+np.array([2,-1,.5]))))<2e-6

def test_six_model_complete_family():
    pairs=[tuple(itertools.combinations([i for i in range(6) if i!=t],2)) for t in range(6)]
    assert sum(map(len,pairs))==60
    assert all(len(p)==10 for p in pairs)
    assert all(t not in p for t,ps in enumerate(pairs) for p in ps)

def test_policy_strong_fixed_and_test_rejection():
    rows=[]
    for p in itertools.combinations(range(1,6),2):
      for f in a.FAMILIES:
       for b in (-2,-1)+r.CAPS:
        loss=.1+(.03 if p==(1,2) and b<=2048 else 0)+(.01 if p!=(1,2) else 0)
        rows.append({'pair':list(p),'family':f,'q':2,'codec':'q8','rank':8,'lambda_index':0,'actual_bits':216,'validation_tv':loss,'state_digest':str((p,f,b)),'budget_bits':b,'selection_kind':'pair_family','status':'eligible'})
    policies=r.select_policies(rows,0,2)
    reverse=r.select_policies(list(reversed(rows)),0,2)
    assert policies==reverse
    for scope in r.SCOPES:
      for name in r.POLICIES:
        assert len({tuple(x['pair']) for x in policies if x['scope']==scope and x['policy']==name})==1
    with pytest.raises(ValueError):r.select_policies([x|{'test_tv':0} for x in rows],0,2)

def test_fresh_manifest_sealed_bank_end_to_end(tmp_path):
    import json,subprocess
    import pandas as pd
    rng=np.random.default_rng(94);mf=[];records=[]
    for i in range(40):
        role='fit' if i<20 else ('validation' if i<28 else 'iid_test')
        row={'task_id':'academic_mcq4','item_id':f'new_{i}','prompt_hash':f'prompt_{i}','item_cluster':f'cluster_{i}','seed':81724031,'split':role}
        mf.append(row)
        shared=rng.normal(size=3)
        for model in r.MODELS:
            coords=shared+rng.normal(size=3)*.2
            records.append(row|{'model_id':model,'revision':r.REVISIONS[model]}|{f'coord_{j}':coords[j] for j in range(3)})
    manifest=tmp_path/'splits';bank=tmp_path/'development';sealed=tmp_path/'sealed';out=tmp_path/'run'
    manifest.mkdir();bank.mkdir()
    pd.DataFrame(mf).to_parquet(manifest/'split_seed_81724031.parquet',index=False)
    data=pd.DataFrame(records);data[data.split!='iid_test'].to_parquet(bank/'six.parquet',index=False)
    dims=tmp_path/'dims.json';dims.write_text(json.dumps({'academic_mcq4':3}))
    cmd=[sys.executable,str(ROOT/'experiments/phase_a_run.py'),'--manifest-dir',str(manifest),'--response-dir',str(bank),'--test-response-dir',str(sealed),'--dimensions-json',str(dims),'--output',str(out),'--targets','0','--q','2','--workers','1','--bootstrap-draws','100','--data-role','synthetic_test_only']
    subprocess.run(cmd+['--stage','fit'],check=True,capture_output=True)
    globalfreeze=json.loads((out/'all_selections_frozen.json').read_text())
    assert globalfreeze['cells']==1 and not (out/'evaluation_inputs.json').exists()
    sealed.mkdir();data[data.split=='iid_test'].to_parquet(sealed/'six.parquet',index=False)
    evaluated=subprocess.run(cmd+['--stage','evaluate'],capture_output=True,text=True)
    assert evaluated.returncode==0,evaluated.stdout+evaluated.stderr
    result=pd.read_csv(out/'policy_gains.csv')
    assert set(r.POLICIES)<=set(result.comparator) and set(result.scope)==set(r.SCOPES)
    assert (out/'complete.json').exists()
