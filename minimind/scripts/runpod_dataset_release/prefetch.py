"""Overlap local downloads with publication; dataset locks prevent concurrent writes."""
import time
from publish import SSH,RELEASE,remote
from transfer import download
for kind in ['sft','vision']:
    deadline=time.time()+7200
    while remote(f'test -f /tmp/dataset-release/{kind}/SHA256SUMS.json && echo ready || echo waiting').strip()!='ready':
        if time.time()>deadline:raise TimeoutError(kind)
        time.sleep(20)
    print('DOWNLOAD',kind,flush=True)
    download(SSH,kind,RELEASE/'datasets'/kind)
    print('VERIFIED',kind,flush=True)
