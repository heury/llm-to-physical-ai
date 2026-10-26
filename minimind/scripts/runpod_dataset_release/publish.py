"""Resume-safe local export transfer, public release and anonymous integrity checks.
Credentials remain local. Never deletes the temporary Pod until all three releases verify.
"""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
os.environ.setdefault('HF_XET_HIGH_PERFORMANCE','1')
from huggingface_hub import HfApi,hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError
from cards import BASE,RELEASE,REPOS,attach
from transfer import download

SSH=['ssh','-i',str(Path.home()/'.runpod/ssh/runpodctl-ssh-key'),'-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=30','-p','21066','root@213.173.105.108']
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def log(**kw):
    kw['timestamp']=time.time();print(json.dumps(kw),flush=True);write(RELEASE/'status.json',kw)
def remote(cmd):return subprocess.run(SSH+[cmd],check=True,text=True,capture_output=True).stdout

def verify(api,repo,commit,manifest,dest):
    public=HfApi(token=False);info=public.dataset_info(repo,revision=commit,files_metadata=True);assert not info.private and info.sha==commit
    siblings={s.rfilename:s for s in info.siblings}
    checks={}
    for name,expected in manifest.items():
        s=siblings[name];assert s.size==expected['bytes'],name
        if s.lfs:
            assert s.lfs.sha256==expected['sha256'],name;checks[name]='public LFS SHA256 and byte size match'
        else:
            p=hf_hub_download(repo,name,repo_type='dataset',revision=commit,token=False,cache_dir=str(RELEASE/'verify-cache'))
            assert sha(p)==expected['sha256'];checks[name]='anonymous download SHA256 and byte size match'
    # Independently download one public data shard, not just Hub metadata.
    sample=next(n for n in manifest if n.endswith('.parquet'))
    p=hf_hub_download(repo,sample,repo_type='dataset',revision=commit,token=False,cache_dir=str(RELEASE/'verify-cache'))
    assert sha(p)==manifest[sample]['sha256'];checks[sample]='anonymous Parquet download SHA256 match'
    return checks

def main():
    api=HfApi();assert api.whoami()['name']=='caspar'
    for kind,repo in REPOS.items():
        receipt=RELEASE/f'{kind}-published.json'
        if receipt.exists():continue
        log(phase='waiting_for_export',dataset=kind)
        deadline=time.time()+7200
        while remote(f'test -f /tmp/dataset-release/{kind}/SHA256SUMS.json && echo ready || echo waiting').strip()!='ready':
            if 'Traceback' in remote('tail -30 /tmp/dataset-export.log'):raise RuntimeError(remote('tail -30 /tmp/dataset-export.log'))
            if time.time()>deadline:raise TimeoutError(kind)
            time.sleep(20)
        dest=RELEASE/'datasets'/kind;dest.mkdir(parents=True,exist_ok=True)
        log(phase='download_export',dataset=kind)
        exported=download(SSH,kind,dest)
        attach(kind,dest)
        manifest={str(p.relative_to(dest)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json' and '.cache' not in p.parts}
        write(dest/'SHA256SUMS.json',manifest)
        complete=dict(manifest);complete['SHA256SUMS.json']=dict(bytes=(dest/'SHA256SUMS.json').stat().st_size,sha256=sha(dest/'SHA256SUMS.json'))
        own=RELEASE/f'{kind}-created.json'
        if not own.exists():
            try:api.dataset_info(repo)
            except RepositoryNotFoundError:pass
            else:raise RuntimeError(f'Refusing to overwrite unowned repository: {repo}')
            api.create_repo(repo,repo_type='dataset',private=False)
            write(own,dict(repo=repo,created_at=time.time()))
        log(phase='upload',dataset=kind,repo=repo,bytes=sum(x['bytes'] for x in manifest.values()))
        result=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=dest,commit_message='Release exact completed training data v1.0',ignore_patterns=['.cache/**'])
        commit=result.oid
        log(phase='anonymous_verification',dataset=kind,commit=commit)
        checks=verify(api,repo,commit,complete,dest)
        api.create_tag(repo,tag='v1.0',revision=commit,repo_type='dataset',exist_ok=True)
        assert HfApi(token=False).dataset_info(repo,revision='v1.0').sha==commit
        write(receipt,dict(repo=repo,commit=commit,tag='v1.0',private=False,counts=json.loads((dest/'export-counts.json').read_text()),bytes=sum(x['bytes'] for x in complete.values()),verified_at=time.time(),verification=checks))
        log(phase='published',dataset=kind,repo=repo,commit=commit)
    log(phase='all_published')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        log(phase='failed',error=repr(e));raise
