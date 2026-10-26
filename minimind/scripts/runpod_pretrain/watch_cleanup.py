"""Read-only local watcher: confirm eventual task Pod deletion independently."""
import json,time
from pathlib import Path
import requests
REPO=Path(__file__).resolve().parents[2]
OUT=REPO.parent/'outputs/minimind/pretrain-rtx5090-20261023'
OUT.mkdir(parents=True,exist_ok=True)
cfg={}
for line in (REPO.parent/'.env').read_text().splitlines():
 if '=' in line and not line.lstrip().startswith('#'):
  k,v=line.split('=',1);cfg[k.strip()]=v.strip().strip('\"\'')
s=requests.Session();s.headers['Authorization']='Bearer '+cfg['RUNPOD-API-KEY']
for _ in range(2880):
 try:
  r=s.get('https://api.runpod.io/v2/pods/g1ol0tfb8yk3ik',timeout=30)
  if r.status_code==404:
   (OUT/'cleanup_verified.json').write_text(json.dumps(dict(pod_id='g1ol0tfb8yk3ik',confirmed_absent=True,timestamp=time.time(),global_volume_id='cmuxpmy6p000007l12uzr990b'),indent=2))
   break
  r.raise_for_status()
  p=r.json();(OUT/'pod_status.json').write_text(json.dumps(dict(pod_id=p['id'],status=p['status'],cost=p.get('cost'),timestamp=time.time()),indent=2))
 except requests.RequestException:
  pass
 time.sleep(60)
