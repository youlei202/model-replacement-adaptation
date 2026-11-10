import ast,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.confirmation_prepare import signature,cluster_union
from experiments.exposure_audit import normalize_text
from experiments.confirmation_collect import score

def test_signature_exactly_matches_original_source():
 source=ast.parse((ROOT/'reference_implementation/kernel_isqed/data.py').read_text())
 nodes=[node for node in source.body if isinstance(node,(ast.FunctionDef,ast.Import,ast.ImportFrom)) and (not isinstance(node,ast.FunctionDef) or node.name in ['normalize_text','_shingles','_signature'])]
 funcs=[node for node in nodes if isinstance(node,ast.FunctionDef)]
 scope={'hashlib':hashlib,'re':__import__('re'),'Set':set,'Tuple':tuple}
 exec(compile(ast.Module(body=funcs,type_ignores=[]),'<original-dedup>','exec'),scope)
 for text in ['Short Q?','The brown fox jumped over a sleepy dog, and then walked home.',' '.join(f'word{i}' for i in range(150))]:
  assert tuple(signature(text))==scope['_signature'](scope['_shingles'](text))

def test_global_clustering_blocks_context_variants_and_preserves_old_clusters(tmp_path):
 rows=[dict(task_id='old',item_id='old1',context='What is the capital of France?',options=json.dumps(['Paris','London']),historical=True,item_cluster='oldcluster'),dict(task_id='old',item_id='old2',context='Different historical text',options=json.dumps(['A','B']),historical=True,item_cluster='oldcluster'),dict(task_id='new',item_id='new1',context='What is the capital of France?',options=json.dumps(['Paris','London','Berlin','Madrid']),historical=False,item_cluster=''),dict(task_id='new2',item_id='new2',context='A unique independent next question about calculus',options=json.dumps(['1','2']),historical=False,item_cluster='')]
 out,edges=cluster_union(pd.DataFrame(rows),tmp_path/'cache.npz',workers=1)
 assert out.iloc[2].historical_cluster_overlap
 assert out.iloc[0].item_cluster==out.iloc[1].item_cluster==out.iloc[2].item_cluster
 assert not out.iloc[3].historical_cluster_overlap
 assert (edges.match_rule=='exact_context').any()

def test_new_collector_semantic_permutation_and_dimension():
 class Fake:
  def score(self,system,user,codes):
   assert codes==['0','1','2','3'];return SimpleNamespace(code_token_lengths=(1,1,1,1),log_scores=np.array([10.,20.,30.,40.]))
 row=SimpleNamespace(contrast_dim=3,system_prompt='sys',user_prompt='user',permutation='[2, 0, 3, 1]')
 result=score(Fake(),row)
 from kernel_isqed.response import scores_to_coordinates
 np.testing.assert_array_equal(result,scores_to_coordinates(np.array([20.,40.,10.,30.]))[0])

def test_shared_cluster_bootstrap_preserves_target_dependence():
 from experiments.confirmation_analyze import bootstrap_joint
 a=np.arange(30,dtype=float)/1000
 ci=bootstrap_joint(np.stack([a,2*a,np.zeros_like(a)],axis=1),draws=1000,seed=7,alpha=.05,comparisons=8)
 np.testing.assert_allclose(ci[:,1],2*ci[:,0],rtol=0,atol=1e-15)
 np.testing.assert_array_equal(ci[:,2],0)

def test_frozen_cohort_no_historical_or_role_overlap():
 base=ROOT/'experiments/runs/v2/confirmation'
 if not (base/'data_lock.json').is_file():
  import pytest
  pytest.skip('External historical confirmation artifacts are required for this integrity check')
 lock=json.loads((base/'data_lock.json').read_text())
 hist=pd.read_parquet(base/'global_duplicate_registry.parquet');old=set(hist.loc[hist.historical,'item_cluster'])
 for seed in lock['training_seeds']:
  d=pd.read_parquet(base/'manifests'/f'split_seed_{seed}.parquet')
  assert not (set(d.item_cluster)&old)
  assert d.groupby('item_cluster').split.nunique().max()==1
  assert not d.duplicated(['task_id','item_id']).any()
 for rel,expected in lock['input_hashes'].items():
  assert hashlib.sha256((base/rel).read_bytes()).hexdigest()==expected

def test_collector_refuses_test_before_method_and_resource_lock(tmp_path,monkeypatch):
 from experiments.confirmation_collect import main
 import pytest
 manifest=tmp_path/'required_items_test.parquet';pd.DataFrame({'task_id':['synthetic_guard_only'],'item_id':['unqueried']}).to_parquet(manifest,index=False)
 (tmp_path/'data_lock.json').write_text(json.dumps({'input_hashes':{'required_items_test.parquet':hashlib.sha256(manifest.read_bytes()).hexdigest()}}))
 monkeypatch.setattr(sys,'argv',['collector','--model-index','0','--stage','test','--data-dir',str(tmp_path)])
 with pytest.raises(RuntimeError,match='Test collection is prohibited'):main()
 assert not (tmp_path/'response_cache_test').exists()

def test_analysis_uses_primary_seed_and_does_not_multiply_cluster_n(tmp_path,monkeypatch):
 import experiments.confirmation_analyze as analysis
 settings=[{'target_id':name,'target_global_id':i,'task_id':'synthetic_external','actual_test_clusters':10,'sample_size_satisfied':True} for i,name in enumerate(['target_A','target_B'])]
 lock={'settings':settings,'training_seeds':[1,2,3],'primary_training_seed':1,'comparators':analysis.COMPARATORS,'bootstrap_draws':1000,'bootstrap_seed':7,'familywise_alpha':.05,'familywise_comparisons':8,'design_gain_tv':.005,'primary_endpoint':'synthetic fixture'}
 (tmp_path/'data_lock.json').write_text(json.dumps(lock));(tmp_path/'confirmation_lock.json').write_text(json.dumps({'analysis_code_sha256':hashlib.sha256(Path(analysis.__file__).read_bytes()).hexdigest()}))
 def fake_cell(base,setting,seed):
  rows=[]
  for comp in analysis.COMPARATORS:
   for b in analysis.BUDGETS:
    for j in range(10):rows.append({'item_id':str(j),'item_cluster':str(j),'difference':.01 if seed==1 else .2,'fixed_loss':.51 if seed==1 else .7,'joint_loss':.5,'comparator':comp,'budget_bits':b,'target_id':setting['target_id'],'task_id':setting['task_id'],'seed':seed,'fixed_state_digest':'synthetic_fixed','joint_state_digest':'synthetic_joint'})
  return pd.DataFrame(rows)
 (tmp_path/'fits/campaign').mkdir(parents=True)
 pd.DataFrame([{'q':2,'scope':'strengthened','comparator':'budget_aware_nnls','budget_bits':1024,'gain_tv':.01}]).to_csv(tmp_path/'fits/campaign/policy_gains.csv',index=False)
 monkeypatch.setattr(analysis,'cell_differences',fake_cell);monkeypatch.setattr(sys,'argv',['analysis','--root',str(tmp_path)]);analysis.main()
 result=pd.read_csv(tmp_path/'primary_confirmation_results.csv');assert len(result)==8;assert result.independent_test_clusters.eq(10).all();np.testing.assert_allclose(result.gain_tv,.01);np.testing.assert_allclose(result.simultaneous_ci_low,.01)
 assert pd.read_csv(tmp_path/'average_budget_training_sensitivity.csv').seed.nunique()==3
