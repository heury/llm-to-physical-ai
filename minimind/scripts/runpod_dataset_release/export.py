"""Export the exact completed experiments; no training or source mutation."""
import collections, concurrent.futures, hashlib, importlib.util, json, shutil, sys, threading, zipfile
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

VOL=Path('/shared-volume'); OUT=Path('/tmp/dataset-release')
ROOTS={'pretrain':VOL/'minimind-full-pretrain-20261024','sft':VOL/'minimind-mixed-sft-20261024','vision':VOL/'minimind-v-llava-short-20261025'}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def copy(a,b,expected=None):
    b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b)
    if expected:assert sha(b)==expected,(a,b)
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def log(**kw):print(json.dumps(kw),flush=True)
class Shards:
    def __init__(self,path,limit=20000):self.path=path;self.limit=limit;self.rows=[];self.n=0;self.count=0
    def add(self,row):
        self.rows.append(row);self.count+=1
        if len(self.rows)>=self.limit:self.flush()
    def flush(self):
        if not self.rows:return
        self.path.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pylist(self.rows),self.path/f'{self.n:05d}.parquet',compression='zstd')
        self.n+=1;self.rows=[]
def common(kind):
    root=ROOTS[kind];dest=OUT/kind;dest.mkdir(parents=True,exist_ok=True)
    for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja','vocab.json','merges.txt']:
        if (root/'tokenizer'/name).exists():copy(root/'tokenizer'/name,dest/'tokenizer'/name)
    for name in ['prepared.json','dataset_manifest.json','filter_report.json','settings.json']:
        if (root/name).exists():copy(root/name,dest/'provenance'/name)
    return root,dest

def pretrain():
    root,dest=common('pretrain');meta=json.loads((root/'prepared.json').read_text());source=VOL/'fineweb-edu-10BT-score-3.5'
    for name in ['manifest.json','pretrain-score-3.5-conversion.json']:copy(source/name,dest/'provenance'/name)
    for split,info in meta['files'].items():copy(root/'data'/f'{split}.bin',dest/'arrays'/f'{split}.bin',info['sha256'])
    copy(root/'data/document-index.jsonl',dest/'provenance/document-index.jsonl')
    writers={s:Shards(dest/'documents'/s) for s in ['train','validation','excluded']};counts=collections.Counter();digest=hashlib.sha256()
    with (source/'pretrain-score-3.5.jsonl').open('rb') as f,(root/'data/document-index.jsonl').open() as idx:
        for i,(line,entry) in enumerate(zip(f,idx,strict=True)):
            digest.update(line);n,split,first,windows,tokens,dropped=json.loads(entry);assert n==i
            row=json.loads(line);row.update(source_row=i,assigned_split=split,first_window=first,windows=windows,tokenized_tokens=tokens,dropped_tokens=dropped)
            writers[split if windows else 'excluded'].add(row);counts[split]+=1
            if i%100000==0:log(stage='pretrain',rows=i)
    assert digest.hexdigest()==meta['source_sha256'];assert sum(counts.values())==meta['source_documents']
    for w in writers.values():w.flush()
    write(dest/'export-counts.json',{k:w.count for k,w in writers.items()})

def sft():
    root,dest=common('sft');meta=json.loads((root/'prepared.json').read_text())
    spec=importlib.util.spec_from_file_location('sft_common','/tmp/release-code/sft_common.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    copy(root/'data/audit.jsonl',dest/'provenance/audit.jsonl');copy(root/'test-records.json',dest/'provenance/test-records.json')
    for name,info in meta['arrays'].items():copy(root/'data'/f'{name}.bin',dest/'arrays'/f'{name}.bin',info['sha256'])
    audit=collections.defaultdict(dict)
    for line in (root/'data/audit.jsonl').open():
        r=json.loads(line);audit[(r['dataset'],'train' if r['split']=='train' else 'test')][r['index']]=r
    writers={name:Shards(dest/'conversations'/name,10000) for name in meta['arrays']}
    for source in ['gsm8k','ultrachat','smoltalk']:
        for split in ['test','train']:
            selected=audit[(source,split)];i=0;found=0
            files=[f for f in meta['files'] if f['source']==source and f['split']==split]
            for info in files:
                path=root/'raw'/source/Path(info['file']).name;assert sha(path)==info['sha256']
                for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
                    for row in batch.to_pylist():
                        if i in selected:
                            record=dict(selected[i]);record['source_component']=str(row.get('source',source));record['messages_json']=json.dumps(mod.normalize(row,source),ensure_ascii=False,separators=(',',':'));writers[record['split']].add(record);found+=1
                        i+=1
            assert found==len(selected);log(stage='sft',source=source,split=split,selected=found)
    for name,w in writers.items():w.flush();assert w.count==meta['stats'][name]['conversations']
    write(dest/'export-counts.json',{k:w.count for k,w in writers.items()})

def image_bytes(path, records):
    local=threading.local();handles=[];lock=threading.Lock()
    def read(record):
        if not hasattr(local,'archive'):
            local.archive=zipfile.ZipFile(path)
            with lock:handles.append(local.archive)
        return record,local.archive.read(record['member'])
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0,len(records),128):
                yield from pool.map(read,records[start:start+128])
    finally:
        for archive in handles:archive.close()

def vision():
    root,dest=common('vision');counts={}
    copy(root/'data/heldout.json',dest/'provenance/heldout.json')
    copy(root/'raw/source_images.json',dest/'provenance/source_images.json')
    copy(root/'vision_encoder/preprocessor_config.json',dest/'provenance/preprocessor_config.json')
    for stage in ['pretrain','sft']:
        rows=json.loads((root/'raw'/f'{stage}.json').read_text());splits=json.loads((root/'data'/f'{stage}-record-splits.json').read_text());images=json.loads((root/'data'/f'{stage}-images.json').read_text())
        for name in [f'{stage}-record-splits.json',f'{stage}-images.json']:copy(root/'data'/name,dest/'provenance'/name)
        copy(root/'raw'/f'{stage}.json',dest/'selected'/f'{stage}.json')
        for path in sorted((root/'data').glob(f'{stage}-*')):
            if path.suffix in ['.npy','.bin']:copy(path,dest/'arrays'/path.name)
        lookup={i:s for s,inds in splits.items() for i in inds};assert len(lookup)==sum(map(len,splits.values()))
        writers={s:Shards(dest/'conversations'/f'{stage}_{s}',10000) for s in splits if splits[s]};needed={r['image'] for r in rows}
        shard_dir=dest/'images'/stage;shard_dir.mkdir(parents=True,exist_ok=True);image_map={};part=0;written=0
        archive=zipfile.ZipFile(shard_dir/f'{part:04d}.zip','w',compression=zipfile.ZIP_STORED)
        try:
            for i,(r,raw) in enumerate(image_bytes(VOL/'minimind-v-llava-20261025/raw'/f'{stage}.zip',images)):
                assert r['key'] in needed
                if written>450_000_000:
                    archive.close();part+=1;written=0;archive=zipfile.ZipFile(shard_dir/f'{part:04d}.zip','w',compression=zipfile.ZIP_STORED)
                archive.writestr(r['key'],raw);written+=len(raw)
                image_map[r['key']]=dict(r,archive=f'images/{stage}/{part:04d}.zip',sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
                if i%5000==0:log(stage='vision_images',dataset=stage,images=i,total=len(images))
        finally:archive.close()
        write(dest/'provenance'/f'{stage}-image-archives.json',image_map)
        for i,r in enumerate(rows):
            if i not in lookup:continue
            record=dict(r,selected_row=i,image_archive=image_map[r['image']]['archive'],pixel_sha256=image_map[r['image']]['pixel_sha256']);writers[lookup[i]].add(record)
        for s,w in writers.items():w.flush();counts[f'{stage}_{s}']=w.count
        assert set(image_map)==needed
    write(dest/'export-counts.json',counts)

if __name__=='__main__':
    for kind in sys.argv[1:] or ['pretrain','sft','vision']:
        globals()[kind]();dest=OUT/kind
        manifest={str(p.relative_to(dest)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
        write(dest/'SHA256SUMS.json',manifest);log(stage='complete',dataset=kind,files=len(manifest),bytes=sum(v['bytes'] for v in manifest.values()))
