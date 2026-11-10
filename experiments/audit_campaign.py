#!/usr/bin/env python3
"""Read-only audit of historical evidence before the v2 campaign."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import datetime
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / 'experiments/experiment_config'
OUT = ROOT / 'experiments/runs/v2'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    tmp.replace(p)

def main():
    for d in ['audit','protocol','phase_a','confirmation','rich_behavior','resource_benchmark','reports','reproduce']:
        (OUT/d).mkdir(parents=True, exist_ok=True)
    when = datetime.datetime.now(datetime.timezone.utc).isoformat()
    records = []
    for row in json.loads((KIT/'verification/source_inspection.json').read_text())['files']:
        p = ROOT / row['source'].removeprefix('code/')
        actual = sha(p)
        records.append({'path':str(p.relative_to(ROOT)), 'expected_sha256':row['sha256'], 'actual_sha256':actual, 'matches':actual==row['sha256'], 'bytes':p.stat().st_size})
    write(OUT/'audit/source_audit.json', {'timestamp':when,'reference':str(KIT/'verification/source_inspection.json'),'files':records,'drift_count':sum(not r['matches'] for r in records),'migration':'new modules; historical scientific source unchanged'})
    sys.path.insert(0, str(ROOT/'benchmarks'))
    import run_selection as old
    cfg=json.loads((KIT/'configs/campaign.json').read_text())
    models=tuple(r['model_id'] for r in cfg['model_registry'])
    pins={r['model_id']:r['revision'] for r in cfg['model_registry']}
    split=ROOT/'benchmarks/splits'
    split_audit=old.verify_split_protocol(split)
    manifests, manifest_hashes=old.read_manifests(split,None,list(old.DIMENSIONS))
    sources, prov_hashes, issues=old.response_sources(ROOT/'benchmarks/response_cache/recompute_all','complete',models,split,pins)
    assert not issues, issues
    bank, bank_hashes=old.read_responses(sources,models,pins)
    expected=manifests[['task_id','item_id','prompt_hash']].drop_duplicates()
    expected_keys=set(map(tuple,expected.values))
    cache_records=[]
    runtimes=[]
    for model in models:
        part=bank[bank.model_id==model]
        assert set(map(tuple,part[['task_id','item_id','prompt_hash']].values))==expected_keys
        for task,d in old.DIMENSIONS.items():
            assert np.isfinite(part.loc[part.task_id==task,[f'coord_{j}' for j in range(d)]].values).all()
        p=ROOT/'benchmarks/response_cache/recompute_all'/f"{model.replace('/','--')}.provenance.json"
        prov=json.loads(p.read_text())
        runtimes.append(prov['runtime'])
        cache_records.append({'model_id':model,'revision':pins[model],'rows':len(part),'unique_keys':len(part),'task_counts':part.task_id.value_counts().to_dict(),'active_coordinates_finite':True,'official_forward_passed':prov['official_forward_check']['passed'],'runtime':prov['runtime']})
    assert all(r==runtimes[0] for r in runtimes)
    write(OUT/'audit/cache_audit.json',{'timestamp':when,'total_rows':len(bank),'unique_queries':len(expected),'models':cache_records,'split_protocol':split_audit,'manifest_hashes':manifest_hashes,'bank_hashes':bank_hashes,'provenance_hashes':prov_hashes,'identical_runtime':True,'issues':issues})
    write(OUT/'audit/runtime_bridge.json',{'timestamp':when,'phase_a':'No new neural inference: reuse only the six uniform completed recompute_all banks.','cached_runtime':runtimes[0],'historical_mixing':False,'original_to_benchmarks_bridge':'Failed; original response values excluded from new Phase A fitting. Original responses used only as historical exposure/regression evidence.','new_confirmation':'New collector must check official-forward agreement and preserve runtime provenance; no old/new response rows merged.','rich_behavior':'New model family/context/representation; separate response population, not a bridge to MCQ coordinates.'})
    # Hash all previous result artifacts, including the 60 real public contexts.
    preserved=[]
    for base in ['benchmarks/results_complete','benchmarks/splits','benchmarks/response_cache/recompute_all','benchmarks/diagnostics']:
        for p in sorted((ROOT/base).rglob('*')):
            if p.is_file(): preserved.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
    write(OUT/'audit/preserved_historical_artifacts.json',{'timestamp':when,'files':preserved,'bytes':sum(x['bytes'] for x in preserved)})
    baseline=subprocess.check_output(['git','rev-parse','source-baseline'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    (OUT/'audit/repository_map.md').write_text(f'''# Repository audit\n\nThe provided workspace had no Git metadata. Source baseline commit `{baseline}` preserves the pre-v2 code; current branch is `{branch}`. Large historical inputs/results stay on disk and are hash-inventoried separately.\n\n- Original submitted code/data: `reference_implementation/`.\n- First corrected campaign: `benchmarks/`; six complete response banks and all 60 public contexts are available locally. The earlier analysis ZIP omitted those contexts; this workspace does not.\n- Uploaded v2 kit: `experiments/experiment_config/` (reference components, not completed experiments).\n- New implementation: `experiments/*.py`.\n- New outputs: `experiments/runs/v2/`.\n\nSource snapshot drift: {sum(not r['matches'] for r in records)} / {len(records)} checked files. Complete response rows: {len(bank)}; unique historical queries: {len(expected)}.\n\nHistorical response data and old test outcomes are development evidence. Role rotation and old-seed reuse do not produce independent confirmation. The separate exposure registry defines exclusions for new data.\n''')
    write(OUT/'audit/audit_complete.json',{'timestamp':when,'source_files':len(records),'drift_count':sum(not r['matches'] for r in records),'cache_rows':len(bank),'historical_artifacts':len(preserved),'historical_bytes':sum(r['bytes'] for r in preserved),'git_branch':branch,'source_baseline':baseline})
    print(json.dumps(json.loads((OUT/'audit/audit_complete.json').read_text()),indent=2))

if __name__=='__main__':
    main()
