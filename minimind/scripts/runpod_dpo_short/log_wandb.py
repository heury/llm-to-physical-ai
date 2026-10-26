"""Publish captured training metrics from the local host; no keys enter the Pod."""
import json,os
from pathlib import Path
import wandb
ROOT=Path('/home/mind4u/dev/minimind/outputs/minimind/dpo-short-20261025')
for line in Path('/home/mind4u/dev/minimind/.env').read_text().splitlines():
 if line.startswith('WANDB-API-KEY='):os.environ['WANDB_API_KEY']=line.split('=',1)[1].strip().strip('\"\'')
p=ROOT/'wandb-run-id.txt'
if not p.exists():p.write_text(wandb.util.generate_id())
settings=json.loads((ROOT/'artifacts/results/settings.json').read_text());manifest=json.loads((ROOT/'dataset_manifest.json').read_text())
run=wandb.init(project='MiniMind-DPO-Short',name='Dense64M-UltraFeedback-3K-under340',id=p.read_text().strip(),resume='allow',dir=str(ROOT),config={**settings,'base_revision':manifest['base']['revision'],'dataset_revision':manifest['source']['revision'],'split_pairs':{k:v['pairs'] for k,v in manifest['splits'].items()}})
for line in (ROOT/'artifacts/results/metrics.jsonl').read_text().splitlines():
 row=json.loads(line);run.log(row,step=row['step'])
done=json.loads((ROOT/'artifacts/results/completion.json').read_text())
for k in ['before','after','selected_epoch','optimizer_steps','training_seconds','reference_unchanged','changed_parameter_elements']:run.summary[k]=done[k]
url=run.url;run_id=run.id;run.finish()
(ROOT/'wandb.json').write_text(json.dumps({'url':url,'id':run_id,'finished':True},indent=2))
print(url)
