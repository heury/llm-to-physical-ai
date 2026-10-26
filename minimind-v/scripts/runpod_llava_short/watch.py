"""Local durable W&B bridge, final export, public release and task-Pod cleanup."""
import fcntl,json,os,runpy,shlex,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import requests
from common import sha,write
REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'outputs/minimind-v/llava-short-20261025'
REMOTE='/shared-volume/minimind-v-llava-short-20261025'
PID='4d4iep7ygw7otm'
PUBLISH_PYTHON='/tmp/minimind-v-exercise-venv/bin/python'

def ssh(command,**kw):
    d=json.loads((OUT/'pod.json').read_text())['ssh']['direct']
    return subprocess.run(['ssh','-i',str(Path.home()/'.runpod/ssh/runpodctl-ssh-key'),'-o','StrictHostKeyChecking=accept-new','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-p',str(d['port']),d['username']+'@'+d['host'],command],**kw)

def poll():
    script='''import json
from pathlib import Path
r=Path("/shared-volume/minimind-v-llava-short-20261025");d={}
for name in ['status.json','dataset_manifest.json','settings.json','checkpoint.json','smoke_pretrain.json','smoke_sft.json','evaluation_before.json','evaluation_pretrain.json','evaluation_sft.json','evaluation_after.json','completion.json','export_ready.json']:
 p=r/name
 if p.exists():
  try:d[name]=json.loads(p.read_text())
  except (ValueError,FileNotFoundError):pass
for name in ['prepare_exit_code','train_exit_code']:
 p=r/name
 if p.exists():d[name]=p.read_text().strip()
p=r/'metrics.jsonl';d['metrics']=p.read_text().splitlines() if p.exists() else []
if any(d.get(k) not in [None,'0'] for k in ['prepare_exit_code','train_exit_code']):
 d['error_logs']={n:(r/n).read_text()[-6000:] for n in ['prepare.log','train.log'] if (r/n).exists()}
print(json.dumps(d))'''
    r=ssh('python3 -c '+shlex.quote(script),capture_output=True,text=True,timeout=45);r.check_returncode();return json.loads(r.stdout)

def export(receipt):
    archive=OUT/'artifacts.tar.gz'
    with archive.with_suffix('.tmp').open('wb') as f:
        r=ssh('cat '+shlex.quote(receipt['path']),stdout=f,stderr=subprocess.PIPE,timeout=900)
    r.check_returncode();archive.with_suffix('.tmp').replace(archive);assert sha(archive)==receipt['sha256']
    folder=OUT/'bundle';folder.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:tar.extractall(folder,filter='data')
    checks=json.loads((folder/'SHA256.json').read_text())
    assert all(sha(folder/p)==h for p,h in checks.items())
    c=json.loads((folder/'completion.json').read_text());assert c['complete']
    assert sha(folder/'sft_vlm_768.pth')==c['report']['weights_sha256']
    write(OUT/'export_verified.json',dict(sha256=receipt['sha256'],files=len(checks),timestamp=time.time()))
    r=c['report'];samples=json.loads((folder/'samples.json').read_text())
    lines=['# MiniMind-V LLaVA training results','',f"Alignment and SFT complete. Test PPL: {r['test']['perplexity']:.4f}; black-image PPL: {r['test_black_image']['perplexity']:.4f}.",'','Teacher-forced PPL is not visual QA accuracy. Samples are actual unedited outputs; correctness requires image inspection.','']
    for x in samples:lines.extend([f"![test image](bundle/{x['image']})",'','Prompt: '+x['question'],'','Greedy: '+x['greedy']['response'],'','Sampled: '+x['sample']['response'],'','Black-image control: '+x['black']['response'],''])
    (OUT/'report.md').write_text('\n'.join(lines))

def main():
    lock=(OUT/'watch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((OUT/'pod.json').read_text())['id']==PID
    m=runpy.run_path('/tmp/mm-api.py');cfg=m['cfg'];api=m['api'];session=m['s']
    os.environ['WANDB_API_KEY']=cfg['WANDB-API-KEY'];os.environ['WANDB_SILENT']='true'
    import wandb
    meta=OUT/'wandb_run.json'
    if not meta.exists():write(meta,dict(id=wandb.util.generate_id(),project='MiniMind-V-LLaVA-Short'))
    ident=json.loads(meta.read_text())
    run=None
    if not (OUT/'wandb_finished.json').exists():
        run=wandb.init(entity='mind4u',project=ident['project'],id=ident['id'],resume='allow',name='RTX5090-LLaVA-short50K-align1-simpleQA-sft1-20261025',dir=str(OUT),config={'job':'llava-short-20261025','gpu':'RTX5090','length':340,'alignment_epochs':1,'sft_epochs':1})
    cursor_path=OUT/'bridge_cursor.json';cursor=json.loads(cursor_path.read_text())['lines'] if cursor_path.exists() else 0
    while True:
        try:
            write(OUT/'supervisor_heartbeat.json',dict(timestamp=time.time(),pid=os.getpid()))
            if (OUT/'cleanup_verified.json').exists():return
            if not (OUT/'export_verified.json').exists():
                d=poll()
                for name,value in d.items():
                    if name.endswith('.json'):write(OUT/name,value)
                if d.get('status.json',{}).get('phase')=='failed' or any(d.get(k) not in [None,'0'] for k in ['prepare_exit_code','train_exit_code']):
                    write(OUT/'failure.json',d)
                    api('POST','/pods/'+PID+'/action',json={'action':'stop'})
                    state=api('GET','/pods/'+PID);write(OUT/'stopped_on_failure.json',dict(pod_id=PID,status=state['status'],timestamp=time.time()))
                    if run:run.finish(exit_code=1)
                    return
                if run:
                    for i,line in enumerate(d.get('metrics',[])[cursor:],cursor):
                        try:row=json.loads(line)
                        except ValueError:break
                        run.log(row,step=i);cursor=i+1
                    write(cursor_path,dict(lines=cursor))
                if 'export_ready.json' not in d:
                    time.sleep(20);continue
                export(d['export_ready.json'])
            if run:
                done=json.loads((OUT/'bundle/completion.json').read_text())
                run.summary.update({'complete':True,'test_ppl':done['report']['test']['perplexity'],'weights_sha256':done['report']['weights_sha256']});run.finish();run=None
                write(OUT/'wandb_finished.json',dict(id=ident['id'],timestamp=time.time()))
            if not (OUT/'published.json').exists():
                with (OUT/'publish.log').open('a') as f:
                    subprocess.run([PUBLISH_PYTHON,str(Path(__file__).parent/'publish.py')],stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'HF_HUB_DISABLE_PROGRESS_BARS':'1'},check=True,timeout=1800)
            if not (OUT/'wandb_verified.json').exists():
                query='query($id:String!){project(name:"MiniMind-V-LLaVA-Short",entityName:"mind4u"){run(name:$id){state summaryMetrics}}}'
                response=requests.post('https://api.wandb.ai/graphql',auth=('api',cfg['WANDB-API-KEY']),json={'query':query,'variables':{'id':ident['id']}},timeout=30);response.raise_for_status();d=response.json()['data']['project']['run'];assert d['state']=='finished' and json.loads(d['summaryMetrics'])['complete'];write(OUT/'wandb_verified.json',d)
            response=session.delete('https://api.runpod.io/v2/pods/'+PID,timeout=45)
            assert response.ok or response.status_code==404
            response=session.get('https://api.runpod.io/v2/pods/'+PID,timeout=30);assert response.status_code==404
            write(OUT/'cleanup_verified.json',dict(pod_id=PID,confirmed_absent=True,shared_volume_preserved=True,timestamp=time.time()))
            with (OUT/'report.md').open('a') as f:f.write('\nPublic model: '+json.loads((OUT/'published.json').read_text())['url']+'\nTask Pod terminated; shared volume preserved.\n')
            return
        except Exception:
            write(OUT/'watch_error.json',dict(timestamp=time.time(),traceback=traceback.format_exc()))
            time.sleep(30)

if __name__=='__main__':main()
