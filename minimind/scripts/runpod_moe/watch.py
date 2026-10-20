import base64,fcntl,hashlib,json,os,re,shlex,subprocess,tarfile,time,zlib
from pathlib import Path
root='/shared-volume/minimind-moe-fineweb-20261020'
status=Path('/tmp/minimind-moe-fineweb-watch-status.json')
dest=Path('/Users/mind4u/dev/minimind/outputs/minimind/fineweb-moe-20261020')
def record(**values):
    temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(updated=time.time(),**values),ensure_ascii=False,indent=2));temp.replace(status)
def download(remote_archive='/shared-volume/minimind-moe-fineweb-20261020/artifacts.tar.gz'):
    s=json.load(open('/tmp/moe-pod.json'))['ssh']['proxy']
    key=Path('/tmp/minimind-runpod-ssh-key-path').read_text().strip()
    cmd=['ssh','-tt','-i',key,'-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new',
         '-o','UserKnownHostsFile=/tmp/minimind-runpod-known-hosts','-o','ServerAliveInterval=30',s['username']+'@'+s['host']]
    script="import base64,sys; f=open("+repr(remote_archive)+",'rb'); print('__ARTIFACT_BEGIN__',flush=True); [print(base64.b64encode(b).decode(),flush=True) for b in iter(lambda:f.read(3072),b'')]; print('__ARTIFACT_END__',flush=True)"
    payload='stty -echo\npython3 -c '+shlex.quote(script)+'\nexit\n'
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    p.stdin.write(payload.encode());p.stdin.close()
    started=complete=False;total=0
    archive=Path('/tmp/minimind-moe-fineweb-downloaded.tar.gz')
    with archive.open('wb') as f:
        for line in iter(p.stdout.readline,b''):
            if b'__ARTIFACT_BEGIN__' in line and b"print(" not in line:
                started=True;continue
            if not started:continue
            if b'__ARTIFACT_END__' in line:
                complete=True;break
            raw=base64.b64decode(line.strip(),validate=True)
            f.write(raw);total+=len(raw)
            if total%393216==0:record(status='downloading',bytes=total,destination=str(dest))
    p.stdout.read();p.wait(timeout=30)
    assert complete and p.returncode==0
    dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as t:t.extractall(dest,filter='data')
    hashes=json.loads((dest/'SHA256.json').read_text())
    for n,h in hashes.items():
        with (dest/n).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,n
    evaluation=json.loads((dest/'evaluation_after.json').read_text())
    assert hashes['pretrain_768_moe.pth']==evaluation['checkpoint_sha256']
    record(status='complete',destination=str(dest),bytes=total,files_verified=len(hashes),evaluation=evaluation)

POD_ID = 'yj7u51c5os0wng'
POD_NAME = 'minimind-moe-fineweb-b300-20261020'
AUTHORIZATION = Path('/tmp/minimind-moe-fineweb-cleanup-authorization.json')

def verified_local_result():
    authorization = json.loads(AUTHORIZATION.read_text())
    assert authorization == {'pod_id': POD_ID, 'action': 'terminate_after_verified_download'}
    hashes = json.loads((dest/'SHA256.json').read_text())
    required = {'pretrain_768_moe.pth', 'evaluation_after.json', 'dataset_manifest.json', 'training_settings.json'}
    assert required <= hashes.keys()
    for name, expected in hashes.items():
        path = (dest/name).resolve()
        assert path.is_relative_to(dest.resolve())
        with path.open('rb') as file:
            assert hashlib.file_digest(file,'sha256').hexdigest() == expected, name
    evaluation = json.loads((dest/'evaluation_after.json').read_text())
    manifest = json.loads((dest/'dataset_manifest.json').read_text())
    settings = json.loads((dest/'training_settings.json').read_text())
    steps = settings['steps_per_epoch']
    assert evaluation['steps'] == steps and evaluation['epochs'] == settings['epochs']
    assert evaluation['optimizer_updates'] == settings['optimizer_updates']
    assert evaluation['finite_weights'] and evaluation['strict_load'] and evaluation['finite_optimizer']
    assert evaluation['checkpoint_sha256'] == hashes['pretrain_768_moe.pth']
    return evaluation, len(hashes)

def cli(*args):
    result = subprocess.run(['python3','/tmp/minimind-runpod-cli.py',*args],capture_output=True,text=True,timeout=90)
    if result.returncode:
        raise RuntimeError('Runpod read/action failed: '+(result.stderr or result.stdout)[-1000:])
    return json.loads(result.stdout)

def terminate_verified_pod():
    evaluation, count = verified_local_result()
    # A durable receipt allows retrying cleanup without SSH if deletion succeeded
    # but its API response or the subsequent inventory read was interrupted.
    receipt = dest/'verified_download_receipt.json'
    receipt.write_text(json.dumps(dict(pod_id=POD_ID,files_verified=count,evaluation=evaluation,updated=time.time()),indent=2))
    pods = cli('pod','list')
    target = next((pod for pod in pods if pod['id']==POD_ID), None)
    if target is not None:
        assert target['name'] == POD_NAME, 'Task Pod provenance mismatch'
        record(status='terminating_verified_pod',pod_id=POD_ID,files_verified=count,destination=str(dest))
        # Immutable ID, never derived from inventory or other running jobs.
        cli('pod','delete',POD_ID)
        pods = cli('pod','list')
    assert all(pod['id'] != POD_ID for pod in pods), 'Pod deletion not yet confirmed'
    report=dict(status='complete',pod_id=POD_ID,pod_deleted=True,files_verified=count,
                destination=str(dest),shared_volume_preserved=True,evaluation=evaluation)
    (dest/'cleanup.json').write_text(json.dumps(dict(updated=time.time(),**report),indent=2))
    record(**report)

if __name__=='__main__':
    lock=open('/tmp/minimind-moe-fineweb-watch.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    remote_script="""import base64,json,zlib
from pathlib import Path
r=Path('/shared-volume/minimind-moe-fineweb-20261020');d={}
for n in ['dataset_manifest.json','training_settings.json','smoke.json','artifact.json','evaluation_before.json','evaluation_pretrain.json','evaluation_sft.json','evaluation_after.json','wandb_run.json','progress.json','checkpoint.json','phase.json']:
 p=r/n
 if p.exists():d[n]=json.loads(p.read_text())
p=r/'exit_code';d['exit_code']=p.read_text().strip() if p.exists() else None
p=r/'training.log';lines=p.read_text().splitlines() if p.exists() else []
d['last_log']=chr(10).join(lines[-8:])
print('LORA_WATCH:'+base64.b64encode(zlib.compress(json.dumps(d).encode())).decode())
"""
    while True:
        try:
            if (dest/'verified_download_receipt.json').exists():
                terminate_verified_pod();break
            command='python3 -c '+shlex.quote(remote_script)
            r=subprocess.run(['python3','/tmp/moe_proxy.py',command],capture_output=True,text=True,timeout=90)
            if r.returncode:raise RuntimeError(r.stdout[-1000:] or r.stderr[-1000:])
            m=re.search(r'LORA_WATCH:([A-Za-z0-9+/=]+)',r.stdout)
            if not m:raise RuntimeError('Remote status marker missing')
            d=json.loads(zlib.decompress(base64.b64decode(m.group(1))))
            dest.mkdir(parents=True,exist_ok=True)
            for name,value in d.items():
                if name.endswith('.json'):(dest/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
            if d['exit_code']=='0':
                download()
                terminate_verified_pod()
                break
            if d['exit_code'] is not None:
                record(status='training_failed',exit_code=d['exit_code'],phase=d.get('phase.json'),last_log=d['last_log']);break
            record(status='watching',pod_id='yj7u51c5os0wng',destination=str(dest),phase=d.get('phase.json'),wandb=d.get('wandb_run.json'),progress=d.get('progress.json'),last_log=d['last_log'],cleanup_policy='terminate_task_pod_after_verified_final_download')
        except Exception as error:record(status='retry',error=str(error),destination=str(dest))
        time.sleep(60)
