"""Scientific pairing, noninferiority, and unsigned-disagreement regressions."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mcq_resource_analyze import intervals,paired_resource_statistics,align_paired_blocks,quality,sha

def test_identical_systems_are_exact_zero():
    x=np.array([2,200,3,50,8,20,9,100,25,11.],dtype=float)
    d=paired_resource_statistics(x,x,draws=2000)
    assert d['resource_reduction']==d['resource_reduction_low']==d['resource_reduction_high']==0
    assert d['cost_difference_ms_per1000']==d['cost_difference_simultaneous_low']==d['cost_difference_simultaneous_high']==0
    z=intervals(np.zeros((10,3)),comparisons=10,draws=100)
    assert all(np.array_equal(v,np.zeros(3)) for v in z.values())

def test_paired_ratio_keeps_shared_block_slowdowns():
    fixed=np.array([1,1000,3,800,2,600,4,400,8,200.],dtype=float)
    d=paired_resource_statistics(.8*fixed,fixed,draws=2000)
    assert np.isclose(d['resource_reduction'],.2)
    assert np.isclose(d['resource_reduction_low'],.2) and np.isclose(d['resource_reduction_high'],.2)
    a=pd.DataFrame({'mean_end_to_end_ms_per1000':.8*fixed,'GPU_uuid':['gpu_a']*10},index=range(10))
    b=pd.DataFrame({'mean_end_to_end_ms_per1000':fixed,'GPU_uuid':['gpu_a']*10},index=range(10))
    aa,bb,indices,uuids=align_paired_blocks(a,b.sample(frac=1,random_state=4))
    assert np.array_equal(aa,.8*fixed) and np.array_equal(bb,fixed) and uuids==['gpu_a']
    b.loc[0,'GPU_uuid']='gpu_b'
    with pytest.raises(ValueError,match='GPU UUID'):align_paired_blocks(a,b)
    b.loc[0,'GPU_uuid']=''
    with pytest.raises(ValueError,match='GPU UUID'):align_paired_blocks(a,b)
    with pytest.raises(ValueError,match='identical paired'):align_paired_blocks(a,b.iloc[1:])

def test_quality_source_guard_and_argmax_intersection_union(tmp_path):
    catalog=tmp_path/'catalog';catalog.mkdir();cell=tmp_path/'cell';cell.mkdir();out=tmp_path/'out';out.mkdir()
    freeze={'target':1,'seed':81724031,'task':'academic_mcq4','q':2}
    (cell/'selection_freeze.json').write_text(json.dumps(freeze))
    rows=[]
    for i in range(50):
        for digest,tv,arg in [('ref',.2,1),('better',.2,0),('worse',.2,1)]:
            rows.append({'item_id':str(i),'item_cluster':'c'+str(i),'state_digest':digest,'tv_loss':tv,'kl_loss':.1,'argmax_disagreement':arg})
    loss=pd.DataFrame(rows);loss.argmax_disagreement=loss.argmax_disagreement.astype(np.uint8)
    loss.to_parquet(cell/'item_predictions_losses.parquet',index=False)
    (cell/'evaluation_complete.json').write_text(json.dumps({'freeze_sha256':sha(cell/'selection_freeze.json'),'created_unix':20,'outputs':{'item_predictions_losses.parquet':sha(cell/'item_predictions_losses.parquet')}}))
    source={'targets':[1],'seed':81724031,'task':'academic_mcq4','source_freeze_hashes':{str((cell/'selection_freeze.json').resolve()):sha(cell/'selection_freeze.json')}}
    (catalog/'candidate_catalog.json').write_text(json.dumps(source));np.savez(catalog/'validation_losses.npz',x=np.zeros(1))
    eligible=[{'system_id':x,'state_digest':x,'family':'nnls','pair':[0,2],'actual_bits':304} for x in ['ref','better','worse']]
    lock={'candidate_catalog_sha256':sha(catalog/'candidate_catalog.json'),'validation_loss_sha256':sha(catalog/'validation_losses.npz'),'operating_points_frozen_before_test':True,'test_quality_used':False,'created_unix':10,'eligibility':eligible,'decisions':[
        {'target_index':1,'policy':'budget_aware','quality_margin':.005,'reference_system':'ref','selected_system':'better','status':'eligible'},
        {'target_index':1,'policy':'fixed_mean_budget','quality_margin':.005,'reference_system':'better','selected_system':'worse','status':'eligible'}]}
    (catalog/'operating_points_lock.json').write_text(json.dumps(lock))
    result=quality(catalog,[cell],out)
    assert result[0]['argmax_disagreement_gap']==-1 # uint8 subtraction must not wrap to255
    assert result[0]['tv_noninferiority_pass_primary_adjustment'] and result[0]['registered_joint_fidelity_claim_pass']
    assert result[1]['tv_noninferiority_pass_primary_adjustment']
    assert not result[1]['argmax_noninferiority_pass_primary_adjustment'] and not result[1]['registered_joint_fidelity_claim_pass']
    freeze['seed']+=1;(cell/'selection_freeze.json').write_text(json.dumps(freeze))
    with pytest.raises(ValueError,match='freeze hash drift'):quality(catalog,[cell],out)
