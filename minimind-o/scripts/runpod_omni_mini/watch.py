"""Host-side completion guard: verify -> upload -> verify Hub -> delete this Pod."""
import os,sys,time,json,subprocess,hashlib,tarfile,traceback
from pathlib import Path
from huggingface_hub import HfApi
POD='zuq0hsnyvgsmjz';HOST='root@74.2.96.213';PORT='15530'
KEY=str(Path.home()/'.runpod/ssh/runpodctl-ssh-key')
REMOTE='/workspace/omni-mini-20261025'
LOCAL=Path('/home/mind4u/dev/minimind/minimind-o/outputs/omni-mini-20261025')
REPO='caspar/MiniMind-O-Mini-LLaVA-5090-20261025'
SSH=['ssh','-i',KEY,'-p',PORT,'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',HOST]
LOCAL.mkdir(parents=True,exist_ok=True)
def write(obj):
 obj['time']=time.time();tmp=LOCAL/'watch-status.tmp';tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(LOCAL/'watch-status.json')
def remote(cmd):return subprocess.run(SSH+[cmd],capture_output=True,text=True,timeout=90)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 deadline=time.time()+12*3600
 while time.time()<deadline:
  try:
   r=remote(f'cat {REMOTE}/SUCCESS.json 2>/dev/null')
   if r.returncode==0:
    success=json.loads(r.stdout);break
   r=remote(f'cat {REMOTE}/status.json 2>/dev/null')
   write(dict(phase='waiting',pod=POD,remote=json.loads(r.stdout) if r.returncode==0 else None))
  except Exception as e:write(dict(phase='waiting',error=str(e),pod=POD))
  time.sleep(45)
 else:
  subprocess.run(['runpodctl','pod','stop',POD],check=True,capture_output=True)
  write(dict(phase='deadline_stopped',pod=POD));return
 write(dict(phase='download',pod=POD))
 bundle=LOCAL/'artifacts.tar.gz'
 subprocess.run(['scp','-i',KEY,'-P',PORT,HOST+':'+REMOTE+'/artifacts.tar.gz',str(bundle)],check=True,timeout=1800)
 assert sha(bundle)==success['bundle_sha256'],'Bundle hash mismatch'
 with tarfile.open(bundle) as tar:tar.extractall(LOCAL,filter='data')
 folder=LOCAL/'artifacts';hashes=json.loads((folder/'SHA256SUMS.json').read_text())
 for name,expected in hashes.items():assert sha(folder/name)==expected,name
 write(dict(phase='upload',pod=POD,repo=REPO,verified_files=len(hashes)))
 api=HfApi();assert api.whoami()['name']=='caspar'
 api.create_repo(REPO,private=True,exist_ok=True)
 commit=api.upload_folder(repo_id=REPO,folder_path=folder,commit_message='Upload RTX 5090 Omni mini and visual fine-tuning with validation and samples')
 info=api.model_info(REPO,revision=commit.oid,files_metadata=True)
 actual={f.rfilename:f for f in info.siblings}
 for name,expected in hashes.items():
  f=actual[name];assert f.size==(folder/name).stat().st_size,name
  if f.lfs:assert f.lfs.sha256==expected,name
 # Prove the principal checkpoint hash is the remote LFS object hash.
 assert actual['sft_omni_768.pth'].lfs.sha256==success['checkpoint_sha256']
 write(dict(phase='uploaded_verified',repo=REPO,commit=commit.oid,checkpoint_sha256=success['checkpoint_sha256'],pod=POD))
 subprocess.run(['runpodctl','pod','delete',POD],check=True,capture_output=True)
 r=subprocess.run(['runpodctl','pod','list'],check=True,capture_output=True,text=True)
 pods=json.loads(r.stdout)
 if isinstance(pods,dict):pods=pods.get('pods',pods.get('data',[]))
 assert all(p.get('id')!=POD for p in pods),'Pod still listed'
 write(dict(phase='complete',repo=REPO,commit=commit.oid,checkpoint_sha256=success['checkpoint_sha256'],pod_deleted=True,verified_files=len(hashes)))
if __name__=='__main__':
 try:main()
 except BaseException:
  write(dict(phase='error',traceback=traceback.format_exc(),pod=POD));raise
