"""Parallel SSH range reads; credentials and source files remain unchanged."""
import concurrent.futures,fcntl,hashlib,json,shlex,subprocess
from pathlib import Path

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def download(ssh,kind,dest,workers=6):
    dest.mkdir(parents=True,exist_ok=True)
    with (dest.parent/("."+kind+"-transfer.lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _download(ssh,kind,dest,workers)

def _download(ssh,kind,dest,workers=6):
    remote=f'/tmp/dataset-release/{kind}'
    raw=subprocess.run(ssh+[f'cat {shlex.quote(remote+"/SHA256SUMS.json")}'],check=True,capture_output=True).stdout
    manifest=json.loads(raw);tasks=[]
    for name,meta in manifest.items():
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.stat().st_size==meta['bytes'] and sha(target)==meta['sha256']:continue
        if meta['bytes']<8*1024*1024:continue
        with target.open('wb') as f:f.truncate(meta['bytes'])
        for offset in range(0,meta['bytes'],128*1024*1024):tasks.append((name,offset,min(128*1024*1024,meta['bytes']-offset)))
    # Small files in one rsync session, large files in independent SSH byte ranges.
    excludes=[f'--exclude=/{n}' for n,m in manifest.items() if m['bytes']>=8*1024*1024]
    transport=shlex.join(ssh[:-1]);host=ssh[-1]
    def small():subprocess.run(['rsync','-rlt','--partial','-e',transport,*excludes,host+':'+remote+'/',str(dest)+'/'],check=True)
    def chunk(task):
        name,offset,size=task
        cmd=f'dd if={shlex.quote(remote+"/"+name)} bs=1M iflag=skip_bytes,count_bytes skip={offset} count={size} status=none'
        with (dest/name).open('r+b',buffering=0) as f:
            f.seek(offset);subprocess.run(ssh+[cmd],stdout=f,check=True)
            assert f.tell()==offset+size,(name,offset,size,f.tell())
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(small)]+[pool.submit(chunk,t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):future.result()
    for name,meta in manifest.items():assert (dest/name).stat().st_size==meta['bytes'] and sha(dest/name)==meta['sha256'],name
    return manifest
