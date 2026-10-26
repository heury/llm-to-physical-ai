"""Persistent local supervisor: live W&B, provision MoE after prep, export and cleanup.
User credentials remain on this local machine. Run under the named user systemd unit.
"""
import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
import requests
import wandb
REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'outputs/minimind/full-pretrain-20261024'
ROOT='/shared-volume/minimind-full-pretrain-20261024'
PROJECT='MiniMind-FineWeb-Full-Pretrain'
cfg={}
for line in (REPO/'.env').read_text().splitlines():
 if '=' in line and not line.lstrip().startswith('#'):
  k,v=line.split('=',1);cfg[k.strip()]=v.strip().strip('\"\'')
os.environ.update(WANDB_API_KEY=cfg['WANDB-API-KEY'],WANDB_MODE='online',WANDB_CACHE_DIR=str(OUT/'wandb-cache'))
s=requests.Session();s.headers['Authorization']='Bearer '+cfg['RUNPOD-API-KEY']
handles={}
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False));temp.replace(path)
def load(path,default=None):return json.loads(Path(path).read_text()) if Path(path).exists() else default
def api(method,path,**kw):
 r=s.request(method,'https://api.runpod.io/v2'+path,timeout=40,**kw)
 if r.status_code==404:return None
 r.raise_for_status();return r.json() if r.content else {}
def ssh(p,command,**kw):
 d=p['ssh']['direct'];return subprocess.run(['ssh','-o','StrictHostKeyChecking=accept-new','-o','BatchMode=yes','-o','ConnectTimeout=15','-i',str(Path.home()/'.runpod/ssh/runpodctl-ssh-key'),'-p',str(d['port']),d['username']+'@'+d['host'],command],timeout=180,**kw)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def sanitized(p):return {k:p.get(k) for k in ['id','name','image','gpu','cost','dataCenterId','mounts','ssh','status']}
def provision_moe():
 record=OUT/'moe/pod.json'
 if record.exists() or not (OUT/'prepared.json').exists():return
 catalog=api('GET','/catalog/gpus',params={'include':'AVAILABILITY','product':'POD'})
 gpu=next(g for g in catalog['gpus'] if g['id']=='NVIDIA GeForce RTX 5090')
 assert gpu['price']['secure']<=1.19,'Price increased beyond quoted $1.19/hr'
 centers=[d['id'] for d in gpu.get('dataCenters',[]) if d['availability']!='NONE']
 if not centers:save(OUT/'capacity.json',dict(waiting=True,timestamp=time.time()));return
 templates=api('GET','/catalog/templates',params={'type':'official'});template=next(t for t in templates['templates'] if t['id']=='runpod-torch-v280')
 body=dict(name='minimind-full-pretrain-moe-20261024',image=template['image'],cloud='SECURE',gpu=dict(id=gpu['id'],count=1,minCudaVersion='12.8',minVcpuCountPerGpu=8),disk=80,ports=['22/tcp'],startSsh=True,startJupyter=False,dataCenterIds=centers,mounts={'global':[{'volumeId':'cmuxpmy6p000007l12uzr990b','path':'/shared-volume'}]})
 p=api('POST','/pods',json=body);save(record,sanitized(p));fresh=api('GET','/pods/'+p['id']);assert fresh['mounts']==body['mounts'] and fresh['name']==body['name'];save(record,sanitized(fresh))
 print('MOE_CREATED',p['id'],flush=True)
def launch_moe(p):
 if (OUT/'moe/launched.json').exists():return
 command='python3 -c \'import torch; x=torch.randn(128,128,device="cuda",requires_grad=True); (x@x).sum().backward(); print(torch.cuda.get_device_name(),"CUDA_BACKWARD_OK")\''
 r=ssh(p,command,capture_output=True,text=True)
 if r.returncode:raise RuntimeError('MoE CUDA preflight: '+r.stderr[-1500:])
 with (OUT/'source.tar.gz').open('rb') as f:
  r=ssh(p,'mkdir -p /tmp/minimind-full-pretrain-source /tmp/minimind-full-pretrain && tar -xz -C /tmp/minimind-full-pretrain-source',stdin=f,capture_output=True);assert r.returncode==0
 r=ssh(p,'setsid bash /tmp/minimind-full-pretrain-source/scripts/runpod_full_pretrain/bootstrap.sh moe > '+ROOT+'/bootstrap-moe.log 2>&1 < /dev/null & echo LAUNCHED',capture_output=True,text=True);assert r.returncode==0
 save(OUT/'moe/launched.json',dict(pod_id=p['id'],cuda_preflight=True,timestamp=time.time()))
def poll(p,kind):
 offset=load(OUT/kind/'bridge_cursor.json',{}).get('bytes',0)
 code='''import json
from pathlib import Path
root=Path(%r);out=root/%r;data={}
for n in ['status.json','training_settings.json','wandb_run.json','checkpoint.json','evaluation_before.json','export_ready.json']:
 p=out/n
 if p.exists():data[n]=json.loads(p.read_text())
for n in ['prepared.json','prepare_status.json']:
 p=root/n
 if p.exists():data[n]=json.loads(p.read_text())
p=out/'metrics.jsonl';rows=[];offset=%d
if p.exists():
 with p.open() as f:
  f.seek(offset)
  for line in f:
   if line.strip():rows.append(json.loads(line))
  offset=f.tell()
data['metrics']=rows;data['offset']=offset
print(json.dumps(data))
'''%(ROOT,kind,offset)
 r=ssh(p,"python3 - <<'PY'\n"+code+'\nPY',capture_output=True,text=True)
 if r.returncode:raise RuntimeError('SSH poll failed: '+r.stderr[-1000:])
 return json.loads(r.stdout)
def live_bridge(kind,payload):
 metadata=payload.get('wandb_run.json');settings=payload.get('training_settings.json')
 if not metadata or not settings:return
 if kind not in handles:
  directory=OUT/'wandb-live';directory.mkdir(exist_ok=True)
  handles[kind]=wandb.init(entity='mind4u',project=PROJECT,id=metadata['id'],resume='allow',name=f'RTX5090-{kind}-full-doc-drop-tail-2epoch-20261024',config=settings,dir=str(directory),mode='online',reinit='create_new',settings=wandb.Settings(disable_git=True))
 run=handles[kind]
 for row in payload.get('metrics',[]):
  run.log({**row,'source_training_timestamp':row['timestamp']})
 save(OUT/kind/'bridge_cursor.json',dict(bytes=payload['offset'],wandb_id=metadata['id'],timestamp=time.time()))
def verify_wandb(kind):
 folder=OUT/kind;complete=load(folder/'completion.json');metadata=load(folder/'wandb_run.json')
 if kind not in handles:
  settings=load(folder/'training_settings.json');handles[kind]=wandb.init(entity='mind4u',project=PROJECT,id=metadata['id'],resume='allow',config=settings,dir=str(OUT/'wandb-live'),mode='online',reinit='create_new',settings=wandb.Settings(disable_git=True))
 run=handles.pop(kind);run.summary.update(complete);run.finish()
 q='query($id:String!){project(name:"'+PROJECT+'",entityName:"mind4u"){run(name:$id){name state summaryMetrics}}}'
 for attempt in range(12):
  r=requests.post('https://api.wandb.ai/graphql',auth=('api',cfg['WANDB-API-KEY']),json={'query':q,'variables':{'id':metadata['id']}},timeout=30);r.raise_for_status();project=r.json().get('data',{}).get('project');result=project.get('run') if project else None
  if result:
   summary=json.loads(result['summaryMetrics'])
   if result['state']=='finished' and summary.get('complete') and summary.get('optimizer_updates')==complete['optimizer_updates']:
    save(folder/'wandb_server_verified.json',result);return
  time.sleep(5)
 raise RuntimeError('W&B final server verification pending')
def export_cleanup(p,kind,receipt):
 folder=OUT/kind;archive=folder/'results.tar.gz'
 if not (folder/'export_verified.json').exists():
  with archive.open('wb') as f:r=ssh(p,'cat '+receipt['archive'],stdout=f,stderr=subprocess.PIPE)
  assert r.returncode==0 and sha(archive)==receipt['sha256']
  with tarfile.open(archive) as tar:tar.extractall(folder,members=[m for m in tar.getmembers() if not(m.issym() or m.islnk())],filter='data')
  files=load(folder/'artifact_manifest.json');assert all(sha(folder/name)==digest for name,digest in files.items())
  c=load(folder/'completion.json');assert c['complete'] and c['epochs']==2 and c['checkpoint_verified'] and c['tests_complete']
  save(folder/'export_verified.json',dict(pod_id=p['id'],sha256=receipt['sha256'],timestamp=time.time()))
 if not (folder/'wandb_server_verified.json').exists():verify_wandb(kind)
 api('DELETE','/pods/'+p['id']);assert api('GET','/pods/'+p['id']) is None
 save(folder/'cleanup_verified.json',dict(pod_id=p['id'],confirmed_absent=True,global_volume_preserved=True,wandb_server_verified=True,timestamp=time.time()))
 print('COMPLETE_POD_TERMINATED',kind,p['id'],flush=True)
def report():
 models={k:load(OUT/k/'completion.json') for k in ['dense','moe']}
 if not all(models.values()):return
 lines=['# Full-document pretraining results','', '| Model | Epochs | Updates | Before PPL | After PPL |','|---|---:|---:|---:|---:|']
 for k,c in models.items():lines.append(f"| {k} | {c['epochs']} | {c['optimizer_updates']} | {c['validation_before']['perplexity']:.4f} | {c['validation_after']['perplexity']:.4f} |")
 for k,c in models.items():lines+=['',k+': '+c['weights'],'',c['wandb_url']]
 (OUT/'report.md').write_text('\n'.join(lines)+'\n');save(OUT/'completion.json',dict(complete=True,models=models,all_task_pods_terminated=True,timestamp=time.time()))
 scenario=REPO/'test_scenarios/minimind/004-full-document-pretrain.md'
 if scenario.exists():
  text=scenario.read_text().replace('상태: 실행 중. 아직 학습 완료 또는 품질 통과 결과가 아님.','상태: 2epoch 학습·체크포인트 검증·W&B 서버 완료·두 작업 Pod 종료 확인 완료. 생성 품질은 별도 평가 대상이다.')
  scenario.write_text(text+'\n완료 수치는 `outputs/minimind/full-pretrain-20261024/report.md`와 각 모델의 완료·종료 기록을 참고한다.\n')
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 for kind in ['dense','moe']:(OUT/kind).mkdir(exist_ok=True)
 while True:
  for kind in ['dense','moe']:
   folder=OUT/kind
   if (folder/'cleanup_verified.json').exists() or (folder/'stopped_on_failure.json').exists():continue
   job=load(folder/'pod.json')
   if not job:continue
   try:
    p=api('GET','/pods/'+job['id'])
    if p is None:raise RuntimeError('Task Pod unexpectedly absent: '+job['id'])
    assert p['name']=='minimind-full-pretrain-'+kind+'-20261024';save(folder/'pod.json',sanitized(p))
    if not (p.get('ssh') or {}).get('direct'):continue
    if kind=='moe':launch_moe(p)
    payload=poll(p,kind)
    for name in ['prepared.json','prepare_status.json']:
     if name in payload:save(OUT/name,payload[name])
    if 'status.json' in payload:save(folder/'live_status.json',payload['status.json'])
    for name in ['training_settings.json','wandb_run.json','checkpoint.json','evaluation_before.json']:
     if name in payload:save(folder/name,payload[name])
    state=payload.get('status.json',{});prep=payload.get('prepare_status.json',{})
    if prep.get('phase') in ['copy_source','tokenizing'] and time.time()-prep.get('timestamp',time.time())>7200:
     prep=dict(prep,phase='failed',detail='Preparation heartbeat stale for over2hours')
    if state and time.time()-state.get('timestamp',time.time())>7200 and state.get('phase') not in ['complete_awaiting_export','waiting_for_data']:
     state=dict(state,phase='failed',detail='No training heartbeat for over2hours; preserving checkpoint and releasing GPU')
    if state.get('phase')=='failed' or prep.get('phase')=='failed':
     save(folder/'failure_detected.json',dict(status=state,prepare=prep,pod_id=p['id'],timestamp=time.time()))
     # Persist diagnostic evidence before releasing this task's GPU on failure.
     log=ssh(p,'tail -n 100 '+ROOT+'/bootstrap-'+kind+'.log',capture_output=True,text=True)
     (folder/'failure.log').write_text(log.stdout+log.stderr);api('POST','/pods/'+p['id']+'/action',json={'action':'stop'});fresh=api('GET','/pods/'+p['id']);save(folder/'stopped_on_failure.json',dict(pod_id=p['id'],read_back_status=fresh['status'],timestamp=time.time()));continue
    live_bridge(kind,payload)
    if payload.get('export_ready.json',{}).get('complete'):export_cleanup(p,kind,payload['export_ready.json'])
   except Exception as e:
    save(folder/'watch_error.json',dict(type=type(e).__name__,detail=str(e),timestamp=time.time()));print('WATCH_ERROR',kind,type(e).__name__,str(e),flush=True)
  try:provision_moe()
  except Exception as e:save(OUT/'provision_error.json',dict(type=type(e).__name__,detail=str(e),timestamp=time.time()))
  save(OUT/'supervisor_heartbeat.json',dict(timestamp=time.time(),pid=os.getpid()))
  if all((OUT/k/'cleanup_verified.json').exists() for k in ['dense','moe']):report();return
  time.sleep(45)
if __name__=='__main__':main()
