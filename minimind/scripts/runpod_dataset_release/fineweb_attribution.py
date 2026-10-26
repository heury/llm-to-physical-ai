"""Preserve original web attribution, joined by the verified source row number."""
import hashlib,json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

def export():
    source=Path('/shared-volume/fineweb-edu-10BT-score-3.5');out=Path('/tmp/dataset-release/pretrain');dest=out/'attribution';dest.mkdir(exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text());index=0;part=0;buffer=[]
    with (source/'pretrain-score-3.5.jsonl').open() as original:
        for shard in manifest['shards']:
            path=source/'data'/shard['output'];digest=hashlib.sha256()
            with path.open('rb') as f:
                for data in iter(lambda:f.read(8*1024*1024),b''):digest.update(data)
            assert digest.hexdigest()==shard['sha256']
            row_index=0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
                for record in batch.to_pylist():
                    expected=json.loads(next(original));assert expected['text']==record.pop('text'),index
                    record.update(source_row=index,source_shard=shard['source'],filtered_shard_row=row_index)
                    buffer.append(record);index+=1;row_index+=1
                    if len(buffer)==40000:
                        pq.write_table(pa.Table.from_pylist(buffer),dest/f'{part:05d}.parquet',compression='zstd');buffer=[];part+=1
            print(json.dumps({'shard':shard['output'],'rows':index}),flush=True)
        assert next(original,None) is None
    if buffer:pq.write_table(pa.Table.from_pylist(buffer),dest/f'{part:05d}.parquet',compression='zstd')
    assert index==1288228
    sums=json.loads((out/'SHA256SUMS.json').read_text())
    for p in dest.glob('*.parquet'):
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        sums[str(p.relative_to(out))]={'bytes':p.stat().st_size,'sha256':digest}
    temp=out/'SHA256SUMS.tmp';temp.write_text(json.dumps(sums,indent=2));temp.replace(out/'SHA256SUMS.json')
    print('ATTRIBUTION_COMPLETE',index,flush=True)
if __name__=='__main__':export()
