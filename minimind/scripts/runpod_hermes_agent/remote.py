"""Local SSH/Runpod helpers; credentials never enter a Pod command."""
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'outputs/minimind/hermes-agent-20261026'
REMOTE='/workspace/minimind-hermes-agent'


def credentials():
    result={}
    for line in (REPO/'.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k,v=line.split('=',1);result[k.strip()]=v.strip().strip('\"\'')
    return result


def control(*args):
    env=os.environ.copy();env['RUNPOD_API_KEY']=credentials()['RUNPOD-API-KEY']
    return subprocess.run([str(Path.home()/'.local/bin/runpodctl'),*args],env=env,capture_output=True,text=True,timeout=120)


def ssh(command,**kwargs):
    p=json.loads((OUT/'pod.json').read_text());info=p['ssh']
    args=['ssh','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(OUT/'known_hosts'),
          '-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30',
          '-i',info['ssh_key']['path'],'-p',str(info['port']),'root@'+info['ip'],command]
    return subprocess.run(args,timeout=kwargs.pop('timeout',180),**kwargs)


if __name__=='__main__':
    r=ssh(sys.argv[1]);sys.exit(r.returncode)
