"""Wait for Secure Cloud RTX 5090 stock, then launch the prepared training pipeline."""
import json
from pathlib import Path
import subprocess
import tarfile
import time
import traceback

from remote import OUT,REPO,ssh,control


def save(name,data):
    p=OUT/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)


def read(name,default=None):
    p=OUT/name;return json.loads(p.read_text()) if p.exists() else default


def get_pod(identifier):
    result=control('pod','get',identifier)
    return json.loads(result.stdout) if not result.returncode else None


def bundle():
    root=REPO/'minimind'
    names=['model/model_minimind.py','model/__init__.py','dataset/lm_dataset.py','dataset/__init__.py',
        'trainer/train_agent.py','trainer/rollout_engine.py','trainer/trainer_utils.py','trainer/__init__.py',
        'scripts/runpod_hermes_agent/prepare.py','scripts/runpod_hermes_agent/evaluate.py','scripts/runpod_hermes_agent/run.py',
        'scripts/runpod_hermes_agent/sft.py','scripts/runpod_hermes_agent/pipeline.py','scripts/runpod_hermes_agent/data.py','scripts/runpod_hermes_agent/bootstrap.sh']
    with tarfile.open(OUT/'source.tar.gz','w:gz') as archive:
        for name in names:
            p=root/name
            if p.exists():archive.add(p,arcname=name)



def launch(active):
    identifier=active['pod_id'];start=time.time()
    while time.time()-start<600:
        pod=get_pod(identifier)
        if pod and pod.get('ssh',{}).get('ip'):
            save('pod.json',pod)
            result=ssh('nvidia-smi --query-gpu=name,memory.total --format=csv',capture_output=True,text=True,timeout=45)
            if result.returncode==0:
                assert '5090' in result.stdout
                save('gpu_verified.json',{'pod_id':identifier,'output':result.stdout});break
        time.sleep(15)
    else:
        control('pod','stop',identifier)
        raise RuntimeError('Task Pod failed SSH readiness within 10 minutes; stopped it')
    bundle()
    with (OUT/'source.tar.gz').open('rb') as f:
        result=ssh('mkdir -p /workspace/minimind-hermes-agent && tar --no-same-owner -xz -C /workspace/minimind-hermes-agent',stdin=f,capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr.decode()[-1000:])
    code='''import subprocess,pathlib
root=pathlib.Path('/workspace/minimind-hermes-agent')
with (root/'bootstrap.log').open('w') as log:
 p=subprocess.Popen(['bash',str(root/'scripts/runpod_hermes_agent/bootstrap.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
'''
    result=ssh("python3 - <<'PY'\n"+code+'\nPY',capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    save('launched.json',{'pod_id':identifier,'volume_id':active['volume_id'],'time':time.time(),'remote_pid':result.stdout.strip()})
    result = subprocess.run(['systemctl','--user','restart','minimind-hermes-agent-20261026.service'])
    if result.returncode:
        subprocess.run(['systemd-run','--user','--unit=minimind-hermes-agent-20261026',
            '--property=WorkingDirectory='+str(REPO),
            '--property=StandardOutput=append:'+str(OUT/'watch.log'),
            '--property=StandardError=append:'+str(OUT/'watch.log'),
            str(REPO/'minimind/.venv/bin/python'),'-u',str(Path(__file__).with_name('watch.py'))],check=True)
    save('provision_status.json',{'stage':'launched','pod_id':identifier,'time':time.time()})
    print('TRAINING_PIPELINE_LAUNCHED',identifier,flush=True)


def main():
    while not read('launched.json'):
        try:
            pending=read('active.json')
            if pending:
                pod=get_pod(pending['pod_id'])
                if pod and pod.get('desiredStatus')=='RUNNING':
                    launch(pending);return
            result=control('gpu','list','--include-unavailable')
            if result.returncode:raise RuntimeError(result.stderr)
            gpu=next(g for g in json.loads(result.stdout) if g.get('gpuId')=='NVIDIA GeForce RTX 5090')
            locations=[s['dataCenterId'] for s in gpu.get('dataCenterAvailability',[]) if s.get('stockStatus','').lower() not in ['none','unavailable','outofstock']]
            save('provision_status.json',{'stage':'waiting_for_rtx5090','locations':locations,'gpu_rate_per_hour':gpu.get('securePricePerHr'),'time':time.time()})
            ledger=read('resources.json')
            locations.sort(key=lambda dc:dc not in ledger['volumes_by_dc'])
            for dc in locations:
                if dc not in ledger['volumes_by_dc']:
                    result=control('network-volume','create','--name','minimind-agent-rl-'+dc.lower(), '--size','50','--data-center-id',dc)
                    if result.returncode:continue
                    volume=json.loads(result.stdout)['id'];ledger['volumes'].append(volume);ledger['volumes_by_dc'][dc]=volume;save('resources.json',ledger)
                volume=ledger['volumes_by_dc'][dc]
                name='minimind-hermes-agent-20261026-'+dc.lower()
                # Recover a successful create whose response was interrupted before its ID was saved.
                listed=control('pod','list','--all')
                existing=[]
                if not listed.returncode:
                    data=json.loads(listed.stdout);pods=data if isinstance(data,list) else data.get('pods',[])
                    existing=[p for p in pods if p.get('name')==name and p.get('desiredStatus')=='RUNNING']
                if existing:
                    identifier=existing[0]['id']
                else:
                    result=control('pod','create','--name',name,'--template-id','runpod-torch-v280',
                        '--gpu-id','NVIDIA GeForce RTX 5090','--gpu-count','1','--min-cuda-version','12.8',
                        '--data-center-ids',dc,'--network-volume-id',volume,'--container-disk-in-gb','40','--ports','22/tcp','--ssh')
                    if result.returncode:
                        print('CAPACITY_WAIT',dc,result.stderr.strip(),flush=True);continue
                    identifier=json.loads(result.stdout)['id']
                if identifier not in ledger['pods']:ledger['pods'].append(identifier)
                save('resources.json',ledger)
                active={'pod_id':identifier,'volume_id':volume,'data_center':dc}
                save('active.json',active)
                launch(active);return
        except Exception as exc:
            print(traceback.format_exc(),flush=True)
            save('provision_error.json',{'error':str(exc),'time':time.time()})
        time.sleep(300)


if __name__=='__main__':main()
