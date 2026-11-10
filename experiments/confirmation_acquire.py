from pathlib import Path
import json,hashlib,datetime
from huggingface_hub import hf_hub_download,HfApi
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'experiments/runs/v2/confirmation';p=out/'population_acquisition_preregistration.json';lock=json.loads(p.read_text());records=[]
for old,info in lock['task_mapping'].items():
 dest=out/'data_sources'/info['external_task_id'];dest.mkdir(parents=True,exist_ok=True)
 for f in [info['file'],'README.md']:
  local=Path(hf_hub_download(info['repository'],filename=f,repo_type='dataset',revision=info['revision'],local_dir=dest))
  records.append({'task_id':info['external_task_id'],'repository':info['repository'],'revision':info['revision'],'file':f,'relative_path':str(local.relative_to(out)),'bytes':local.stat().st_size,'sha256':hashlib.sha256(local.read_bytes()).hexdigest()});print(records[-1],flush=True)
(out/'data_sources_manifest.json').write_text(json.dumps({'preregistration_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'downloaded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':records},indent=2)+'\n')
