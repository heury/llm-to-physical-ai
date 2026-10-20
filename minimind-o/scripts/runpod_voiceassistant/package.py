"""Bundle completed training without chmod/utime on Runpod Global Volumes."""
import hashlib,json,os,shutil,tarfile,time,traceback
from pathlib import Path

def copy_contents(source,destination):
 source=Path(source);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
 for p in source.rglob('*'):
  rel=p.relative_to(source)
  if any(x in {'.cache','__pycache__'} for x in rel.parts):continue
  target=destination/rel
  if p.is_dir():target.mkdir(parents=True,exist_ok=True)
  elif p.is_file():target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def finalize(root,repo):
 import torch
 root=Path(root);repo=Path(repo)
 evaluation=json.loads((root/'evaluation_after.json').read_text())
 manifest=json.loads((root/'dataset_manifest.json').read_text())
 assert evaluation['source_rows']==10000 and manifest['source_rows']==10000
 assert len(manifest['shards'])==40
 weight=root/'out/sft_omni_768.pth'
 assert sha(weight)==evaluation['checkpoint_sha256']
 state=torch.load(weight,map_location='cpu',weights_only=True)
 assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
 resume=torch.load(root/'out/resume.pth',map_location='cpu',weights_only=False)
 assert resume['step']==evaluation['step'] and resume['seen']==evaluation['seen']
 assert all(torch.equal(state[k],resume['model'][k]) for k in state)
 del state,resume
 bundle=repo.parent/'final-bundle'
 if bundle.exists():shutil.rmtree(bundle)
 bundle.mkdir()
 shutil.copyfile(weight,bundle/'sft_omni_768.pth')
 copy_contents(repo,bundle/'source')
 for name in ['tokenizer','selection']:copy_contents(root/name,bundle/name)
 for name in ['SenseVoiceSmall','mimi','vision']:copy_contents(root/'models'/name,bundle/'model'/name)
 for p in list(root.glob('*.json'))+list(root.glob('*.wav'))+[root/'requirements-resolved.txt']:
  if p.name!='artifact.json':shutil.copyfile(p,bundle/p.name)
 hashes={str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()}
 (bundle/'SHA256.json').write_text(json.dumps(hashes,indent=2))
 archive=root/'artifacts.tar.gz';temp=root/'artifacts.tar.gz.tmp'
 with tarfile.open(temp,'w:gz',compresslevel=1) as t:
  for p in bundle.iterdir():t.add(p,arcname=p.name)
 temp.replace(archive)
 (root/'artifact.json').write_text(json.dumps(dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(hashes)),indent=2))
 (root/'SUCCESS').write_text('10000 selected source records processed; final validation, finite strict reload and package passed\n')
 (root/'phase.json').write_text(json.dumps(dict(phase='complete',updated=time.time())))
 (root/'exit_code').write_text('0\n')
 print(json.dumps(dict(status='packaged',files=len(hashes),bytes=archive.stat().st_size)),flush=True)

if __name__=='__main__':
 root=Path('/shared-volume/minimind-o-voiceassistant-20261020')
 try:finalize(root,Path('/tmp/minimind-o-voiceassistant-20261020/repo'))
 except BaseException:
  (root/'exit_code').write_text('1\n');traceback.print_exc();raise
