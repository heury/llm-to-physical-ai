"""Verify W&B server receipt; replay preserved offline history if sync was not visible."""
import argparse,json,os,time
from pathlib import Path
import requests
REPO=Path(__file__).resolve().parents[3]
PROJECT='MiniMind-GSM8K-SFT'
def main(folder):
 folder=Path(folder);cfg={}
 for line in (REPO/'.env').read_text().splitlines():
  if '=' in line and not line.lstrip().startswith('#'):
   k,v=line.split('=',1);cfg[k.strip()]=v.strip().strip('\"\'')
 os.environ['WANDB_API_KEY']=cfg['WANDB-API-KEY'];os.environ['WANDB_MODE']='online'
 metadata=json.loads((folder/'wandb_run.json').read_text());run_id=metadata['id']
 session=requests.Session();session.auth=('api',cfg['WANDB-API-KEY'])
 query='query($id: String!) { project(name: "'+PROJECT+'", entityName: "mind4u") { run(name: $id) { name state summaryMetrics } } }'
 def verify():
  r=session.post('https://api.wandb.ai/graphql',json={'query':query,'variables':{'id':run_id}},timeout=30);r.raise_for_status()
  project=r.json().get('data',{}).get('project')
  result=project.get('run') if project else None
  if result and json.loads(result['summaryMetrics']).get('complete'):
   (folder/'wandb_server_verified.json').write_text(json.dumps(result,indent=2));return True
  return False
 if verify():print('W&B server verified',run_id);return
 import wandb
 from wandb.sdk.internal import datastore
 from wandb.proto import wandb_internal_pb2
 stream=next(folder.rglob('*.wandb'))
 source=datastore.DataStore();source.open_for_scan(str(stream));history=[]
 while True:
  data=source.scan_data()
  if data is None:break
  record=wandb_internal_pb2.Record();record.ParseFromString(data)
  if record.WhichOneof('record_type')=='history':
   row={}
   for item in record.history.item:
    key=item.key or '/'.join(item.nested_key);row[key]=json.loads(item.value_json)
   history.append(row)
 assert len(history)>1
 settings=json.loads((folder/'training_settings.json').read_text());completion=json.loads((folder/'completion.json').read_text())
 runtime=Path('/tmp/minimind-gsm8k-wandb-import');runtime.mkdir(exist_ok=True)
 run=wandb.init(entity='mind4u',project=PROJECT,id=run_id,resume='allow',name='RTX5090-'+settings['model']+'-2epoch-seq340-20261024',dir=str(runtime),mode='online',settings=wandb.Settings(disable_git=True))
 run.config.update(settings,allow_val_change=True)
 with (folder/'history_replayed.jsonl').open('w') as f:
  for row in history:
   f.write(json.dumps(row)+'\n')
   public={k:v for k,v in row.items() if not k.startswith('_')}
   if '_timestamp' in row:public['source_training_timestamp']=row['_timestamp']
   run.log(public,step=int(row.get('_step',0)))
 run.summary.update(completion);run.summary['offline_history_records']=len(history)
 run.finish()
 for _ in range(12):
  if verify():print('W&B server verified after replay',run_id);return
  time.sleep(5)
 raise RuntimeError('W&B server did not confirm uploaded summary')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('folder');a=parser.parse_args();main(a.folder)
