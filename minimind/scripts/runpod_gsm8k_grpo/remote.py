"""SSH only to this task's recorded Pod; no account secrets sent remotely."""
import json
from pathlib import Path
import subprocess
import sys
PROJECT=Path(__file__).resolve().parents[2]
OUT=PROJECT/'outputs/gsm8k-grpo-20261025'
ROOT='/shared-volume/minimind-gsm8k-grpo-20261025'

def ssh(command, **kwargs):
    pod=json.loads((OUT/'pod.json').read_text())
    receipt=OUT/'pod-recovery-created.json'
    if not receipt.exists(): receipt=OUT/'pod-created.json'
    created=json.loads(receipt.read_text())
    assert pod['id']==created['id'] and pod['name']=='minimind-gsm8k-grpo-compare-20261025'
    s=pod['ssh']
    return subprocess.run(['ssh','-o','StrictHostKeyChecking=accept-new','-o','BatchMode=yes','-o','ConnectTimeout=15',
        '-i',s['ssh_key']['path'],'-p',str(s['port']),'root@'+s['ip'],command], **kwargs)

if __name__=='__main__':
    raise SystemExit(ssh(sys.argv[1]).returncode)
