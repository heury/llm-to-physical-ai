"""Local control; credentials remain local and never enter source bundles."""
import json,os,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'outputs/moe-dense-gsm8k-distill-20261025'
def cli(*args):
    env=os.environ.copy()
    for l in (REPO.parent/'.env').read_text().splitlines():
        if l.startswith('RUNPOD-API-KEY='):env['RUNPOD_API_KEY']=l.split('=',1)[1].strip().strip('\"\'')
    r=subprocess.run(['runpodctl',*args],env=env,capture_output=True,text=True);
    if r.returncode:raise RuntimeError(r.stderr)
    return json.loads(r.stdout)
def pod():
    p=cli('pod','get',json.loads((OUT/'pod.json').read_text())['id']);(OUT/'pod-current.json').write_text(json.dumps(p,indent=2));return p

def ssh(command,**kwargs):
    p=pod();d=p['ssh']
    return subprocess.run(['ssh','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(OUT/'known_hosts'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-i',str(Path.home()/'.runpod/ssh/runpodctl-ssh-key'),'-p',str(d['port']),'root@'+d['ip'],command],**kwargs)
if __name__=='__main__':
    if sys.argv[1]=='info':
        p=pod();print(json.dumps({k:p.get(k) for k in ['id','runtimeStatus','cost','ssh','mounts','gpu']}))
    else:sys.exit(ssh(sys.argv[1]).returncode)
