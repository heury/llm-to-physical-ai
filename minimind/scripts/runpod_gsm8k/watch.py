"""Export completed jobs, sync W&B locally, then terminate only the two task Pods."""
import datetime,hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
import requests
REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'outputs/minimind/gsm8k-sft-20261024'
OUT.mkdir(parents=True,exist_ok=True)
ROOT='/shared-volume/minimind-gsm8k-sft-20261024'
cfg={}
for line in (REPO/'.env').read_text().splitlines():
 if '=' in line and not line.lstrip().startswith('#'):
  k,v=line.split('=',1);cfg[k.strip()]=v.strip().strip('\"\'')
s=requests.Session();s.headers['Authorization']='Bearer '+cfg['RUNPOD-API-KEY']
def api(method,path,**kwargs):
 r=s.request(method,'https://api.runpod.io/v2'+path,timeout=30,**kwargs)
 if r.status_code==404:return None
 r.raise_for_status();return r.json() if r.content else {}
def save(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False));t.replace(p)
def ssh(p,cmd,**kwargs):
 direct=p['ssh']['direct'];args=['ssh','-o','StrictHostKeyChecking=accept-new','-o','ConnectTimeout=15','-i',str(Path.home()/'.runpod/ssh/runpodctl-ssh-key'),'-p',str(direct['port']),direct['username']+'@'+direct['host'],cmd]
 return subprocess.run(args,timeout=180,**kwargs)
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def generate_report():
 models=[]
 for kind in ['dense','moe']:
  folder=OUT/kind
  if not (folder/'completion.json').exists():return
  models.append(dict(model=kind,completion=json.loads((folder/'completion.json').read_text()),before=json.loads((folder/'tests_before.json').read_text()),after=json.loads((folder/'tests_after.json').read_text())))
 save(OUT/'comparison.json',dict(complete=True,models=models))
for attempt in range(2880):
 for kind in ['dense','moe']:
  job_file=Path('/tmp/minimind-gsm8k-'+kind+'-pod.json')
  target=OUT/kind;target.mkdir(exist_ok=True)
  if (target/'cleanup_verified.json').exists() or not job_file.exists():continue
  try:
   job=json.loads(job_file.read_text());pod=api('GET','/pods/'+job['id'])
   if pod is None:
    save(target/'unexpected_absence.json',dict(pod_id=job['id'],timestamp=time.time()));continue
   assert pod['name']=='minimind-gsm8k-sft-'+kind+'-20261024'
   if not (pod.get('ssh') or {}).get('direct'):continue
   ready=ssh(pod,'cat '+ROOT+'/'+kind+'/export_ready.json',capture_output=True,text=True)
   if ready.returncode:
    status=ssh(pod,'cat '+ROOT+'/'+kind+'/status.json',capture_output=True,text=True)
    if not status.returncode:save(target/'live_status.json',json.loads(status.stdout))
    continue
   receipt=json.loads(ready.stdout)
   archive=target/'results.tar.gz'
   with archive.open('wb') as f:
    r=ssh(pod,'cat '+receipt['archive'],stdout=f,stderr=subprocess.PIPE)
   if r.returncode:raise RuntimeError('Result download failed')
   assert digest(archive)==receipt['sha256']
   with tarfile.open(archive) as tar:
    # W&B debug logs can link to absolute ephemeral paths; do not extract links.
    members=[m for m in tar.getmembers() if not (m.issym() or m.islnk())]
    tar.extractall(target,members=members,filter='data')
   hashes=json.loads((target/'artifact_manifest.json').read_text())
   assert all(digest(target/name)==sha for name,sha in hashes.items())
   complete=json.loads((target/'completion.json').read_text());assert complete['complete'] and complete['tests_complete'] and complete['test_rows']==1319
   env=os.environ.copy();env['WANDB_API_KEY']=cfg['WANDB-API-KEY'];env['WANDB_MODE']='online'
   offline_runs=[p for p in (target/'wandb').rglob('offline-run-*') if p.is_dir()]
   assert offline_runs, 'No offline W&B run found'
   for offline in offline_runs:
    with (target/'wandb-sync.log').open('a') as log:
     sync=subprocess.run(['/tmp/minimind-sft-tools/bin/wandb','sync','--entity','mind4u','--project','MiniMind-GSM8K-SFT',str(offline)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    if sync.returncode:raise RuntimeError('W&B sync failed')
   with (target/'wandb-verification.log').open('a') as log:
    verified=subprocess.run(['/tmp/minimind-sft-tools/bin/python',str(Path(__file__).with_name('sync_verify.py')),str(target)],stdout=log,stderr=subprocess.STDOUT,timeout=300)
   if verified.returncode:raise RuntimeError('W&B server verification failed')
   # The models and resumable state were verified on the Pod and remain on the Global Volume.
   api('DELETE','/pods/'+job['id'])
   assert api('GET','/pods/'+job['id']) is None
   save(target/'cleanup_verified.json',dict(pod_id=job['id'],confirmed_absent=True,global_volume_preserved=True,wandb_synced=True,wandb_server_verified=True,timestamp=time.time()))
   generate_report()
  except Exception as e:
   save(target/'watch_error.json',dict(type=type(e).__name__,detail=str(e),timestamp=time.time()))
 if all((OUT/k/'cleanup_verified.json').exists() for k in ['dense','moe']):break
 time.sleep(30)
