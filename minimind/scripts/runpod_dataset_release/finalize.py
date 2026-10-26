"""Verify all releases, then delete only this task's temporary CPU Pod."""
import json,os,subprocess,sys,time
from pathlib import Path
import requests
from cards import BASE,RELEASE,REPOS

def main():
    deadline=time.time()+10800
    while not all((RELEASE/f'{kind}-published.json').exists() for kind in REPOS):
        if time.time()>deadline:raise TimeoutError('Publication did not finish; Pod retained')
        time.sleep(20)
    subprocess.run([sys.executable,str(Path(__file__).with_name('validate.py'))],check=True,env={**os.environ,'PYTHONPATH':'/tmp/minimind-pillow-11'})
    created=json.loads((RELEASE/'pod.json').read_text());pod_id=created['id']
    assert pod_id=='23we6vxe00o9yw' and created['name']=='minimind-dataset-release-20261025'
    config={}
    for line in (BASE/'.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k,v=line.split('=',1);config[k.strip()]=v.strip().strip('\"\'')
    session=requests.Session();session.headers['Authorization']='Bearer '+config['RUNPOD-API-KEY']
    url=f'https://api.runpod.io/v2/pods/{pod_id}'
    before=session.get(url,timeout=45);before.raise_for_status()
    assert before.json()['name']==created['name']
    result=session.delete(url,timeout=45);result.raise_for_status()
    after=session.get(url,timeout=45);assert after.status_code==404
    receipt={'pod_id':pod_id,'deleted':True,'readback_status':after.status_code,'verified_at':time.time(),'global_volume_preserved':True}
    (RELEASE/'cleanup.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
