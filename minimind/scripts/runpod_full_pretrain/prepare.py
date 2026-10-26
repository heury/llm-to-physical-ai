"""Tokenize complete documents once; publish verified full chunks, never padding."""
import json,os,shutil,time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from common import *
def main():
 ROOT.mkdir(parents=True,exist_ok=True);RUNTIME.mkdir(exist_ok=True)
 if (ROOT/'prepared.json').exists():return
 manifest=json.loads((SOURCE/'manifest.json').read_text());conversion=json.loads((SOURCE/'pretrain-score-3.5-conversion.json').read_text())
 assert manifest['status']=='complete' and manifest['threshold']==3.5
 assert all(s['min_score']>=3.5 for s in manifest['shards'])
 write(ROOT/'prepare_status.json',dict(phase='copy_source',timestamp=time.time()))
 local=RUNTIME/'source.jsonl';shutil.copyfile(SOURCE/'pretrain-score-3.5.jsonl',local);assert sha(local)==conversion['sha256']
 tok_dir=ROOT/'tokenizer';tok_dir.mkdir(exist_ok=True)
 for p in (SOURCE/'model').iterdir():
  if p.is_file():shutil.copyfile(p,tok_dir/p.name)
 tm=json.loads((SOURCE/'model/dataset_manifest.json').read_text())
 for n,digest in tm.get('tokenizer_sha256',{}).items():assert sha(tok_dir/n)==digest
 tok=AutoTokenizer.from_pretrained(tok_dir,local_files_only=True);assert len(tok)==6400 and (tok.pad_token_id,tok.bos_token_id,tok.eos_token_id)==(0,1,2)
 tok.model_max_length=2**30
 counts={split:dict(documents=0,documents_with_chunks=0,chunks=0,tokenized_tokens=0,retained_unique_tokens=0,dropped_tokens=0) for split in ['train','validation']}
 handles={s:(RUNTIME/f'{s}.bin').open('wb') for s in counts};seen=0;started=time.time()
 index=(RUNTIME/'document-index.jsonl').open('w')
 def process(rows):
  nonlocal seen
  encoded=tok([r['text'] for r in rows],add_special_tokens=False,truncation=False,return_attention_mask=False,return_token_type_ids=False).input_ids
  for row,content in zip(rows,encoded):
   text=row['text'];split=split_for(text);ids=[1]+content+[2];n=chunk_count(len(ids));retained=n*STRIDE+1 if n else 0
   c=counts[split];first=c['chunks'];c['documents']+=1;c['documents_with_chunks']+=n>0;c['chunks']+=n;c['tokenized_tokens']+=len(ids);c['retained_unique_tokens']+=retained;c['dropped_tokens']+=len(ids)-retained
   if n:
    a=np.asarray(ids,dtype=np.uint16);chunks=np.lib.stride_tricks.sliding_window_view(a,LENGTH)[::STRIDE];assert len(chunks)==n
    handles[split].write(chunks.copy().tobytes())
   index.write(json.dumps([seen,split,first,n,len(ids),len(ids)-retained])+'\n');seen+=1
  if seen%8192==0:
   progress=dict(phase='tokenizing',documents_seen=seen,source_documents=conversion['documents'],counts=counts,seconds=time.time()-started,timestamp=time.time());write(ROOT/'prepare_status.json',progress);print(json.dumps(progress),flush=True)
 batch=[]
 with local.open() as f:
  for line in f:
   row=json.loads(line);assert isinstance(row['text'],str);batch.append(row)
   if len(batch)==512:process(batch);batch=[]
 if batch:process(batch)
 for h in handles.values():h.flush();os.fsync(h.fileno());h.close()
 index.close();assert seen==conversion['documents']==manifest['kept']
 data=ROOT/'data';data.mkdir(exist_ok=True);files={}
 for split,c in counts.items():
  assert c['chunks']>0 and c['tokenized_tokens']==c['retained_unique_tokens']+c['dropped_tokens']
  assert (RUNTIME/f'{split}.bin').stat().st_size==c['chunks']*LENGTH*2
  dest=data/f'{split}.bin';shutil.copyfile(RUNTIME/f'{split}.bin',dest);digest=sha(RUNTIME/f'{split}.bin');assert sha(dest)==digest;files[split]=dict(path=str(dest),sha256=digest,bytes=dest.stat().st_size)
 shutil.copyfile(RUNTIME/'document-index.jsonl',data/'document-index.jsonl')
 result=dict(complete=True,source=str(SOURCE),source_sha256=conversion['sha256'],source_revision=manifest['revision'],score_threshold=3.5,score_operator='>= (existing shared-volume filtered dataset)',source_documents=seen,tokenizer_size=6400,max_seq_len=LENGTH,stride=STRIDE,overlap=1,policy='BOS + complete document + EOS; all length340 windows, stride339; drop final short window per document; no padding or cross-document packing; shuffle all train chunks each epoch',split_policy='SHA256(text) first8bytes modulo4096==0 validation, else train; duplicate texts cannot cross splits',splits=counts,files=files,preparation_seconds=time.time()-started,completed_at=time.time())
 write(ROOT/'prepared.json',result);write(ROOT/'prepare_status.json',dict(phase='complete',**result));print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(ROOT/'prepare_status.json',dict(phase='failed',type=type(e).__name__,detail=str(e)));raise
