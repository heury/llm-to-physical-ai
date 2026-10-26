"""Pin UltraChat SFT splits and freeze native assistant-only, length-340 inputs."""
import hashlib,json,os,random,shutil,sys,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dataset.lm_dataset import SFTDataset,pre_processing_chat,post_processing_chat
ROOT=Path('/shared-volume/minimind-ultrachat-sft-20261024')
BASE=Path('/shared-volume/minimind-pretrain-rtx5090-20261023')
REV='8049631c405ae6576f93f445c6b8166f76f5505a'
FILES={'train':['train_sft-00000-of-00003-a3ecf92756993583.parquet','train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet','train_sft-00002-of-00003-ee46ed25cfae92c6.parquet'], 'test':['test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet']}
def save(n,x):
 p=ROOT/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 ROOT.mkdir(exist_ok=True)
 if (ROOT/'prepared.json').exists():return
 tok=AutoTokenizer.from_pretrained(BASE/'tokenizer',local_files_only=True)
 assert len(tok)==6400
 (ROOT/'tokenizer').mkdir(exist_ok=True)
 for source in (BASE/'tokenizer').iterdir():
  if source.is_file():shutil.copyfile(source,ROOT/'tokenizer'/source.name)
 native=SFTDataset.__new__(SFTDataset);native.tokenizer=tok;native.max_length=340
 native.bos_id=tok(tok.bos_token+'assistant\n',add_special_tokens=False).input_ids
 native.eos_id=tok(tok.eos_token+'\n',add_special_tokens=False).input_ids
 manifest=dict(dataset='HuggingFaceH4/ultrachat_200k',revision=REV,max_seq_len=340,split_policy='train_sft only for training; test_sft for evaluation',tokenization='Native SFTDataset chat template and label mask; deterministic frozen native system/think augmentation; first 340 tokens; drop rows with no shifted assistant target',files=[],splits={})
 ids_by_split={}
 for split,files in FILES.items():
  paths=[]
  for f in files:
   local=Path(hf_hub_download(repo_id=manifest['dataset'],repo_type='dataset',filename='data/'+f,revision=REV,cache_dir='/tmp/minimind-ultrachat-hf'))
   raw=ROOT/'raw'/f;raw.parent.mkdir(exist_ok=True);shutil.copyfile(local,raw)
   manifest['files'].append(dict(split=split,name=f,sha256=sha(raw),bytes=raw.stat().st_size));paths.append(local)
  count=sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
  x=np.lib.format.open_memmap('/tmp/'+split+'-input.npy',mode='w+',dtype=np.int32,shape=(count,340))
  y=np.lib.format.open_memmap('/tmp/'+split+'-labels.npy',mode='w+',dtype=np.int32,shape=(count,340))
  kept=dropped=truncated=seen=0;source_ids=set();kept_ids=[]
  for path in paths:
   for chunk in pq.ParquetFile(path).iter_batches(batch_size=512):
    for row in chunk.to_pylist():
     prompt_id=row['prompt_id'];assert prompt_id not in source_ids;source_ids.add(prompt_id)
     messages=row['messages'];assert messages and messages[0]['role']=='user'
     assert all(m['role'] in ['user','assistant','system'] and isinstance(m['content'],str) for m in messages)
     random.seed(42+seen);seen+=1
     prompt=native.create_chat_prompt(pre_processing_chat(messages))
     prompt=post_processing_chat(prompt)
     tokens=tok(prompt,add_special_tokens=False).input_ids
     truncated+=len(tokens)>340
     inputs=tokens[:340]+[tok.pad_token_id]*max(0,340-len(tokens))
     labels=native.generate_labels(inputs)
     assert len(inputs)==len(labels)==340
     if not any(v!=-100 for v in labels[1:]):dropped+=1;continue
     # Every active label is the actual assistant-span token, no user labels.
     assert all(v==-100 or v==inputs[i] for i,v in enumerate(labels))
     x[kept]=inputs;y[kept]=labels;kept+=1;kept_ids.append(prompt_id)
    print(split,'seen',seen,'kept',kept,flush=True)
  assert seen==count and kept>0
  x.flush();y.flush()
  for name,arr in [('input',x),('labels',y)]:
   dest=ROOT/(split+'-'+name+'.npy');tmp=dest.with_name(dest.name+'.tmp')
   with tmp.open('wb') as f:np.save(f,np.asarray(arr[:kept]))
   tmp.replace(dest)
  (ROOT/(split+'-prompt-ids.json')).write_text(json.dumps(kept_ids))
  manifest['splits'][split]=dict(source_rows=count,kept_rows=kept,no_assistant_targets_dropped=dropped,rows_truncated=truncated)
  ids_by_split[split]=source_ids
  del x,y
 assert not (ids_by_split['train'] & ids_by_split['test']), 'Split leakage by prompt id'
 rng=np.random.default_rng(42);test_n=manifest['splits']['test']['kept_rows']
 manifest['validation_indices']=rng.choice(test_n,size=min(512,test_n),replace=False).tolist()
 manifest['arrays']={p.name:sha(p) for p in ROOT.glob('*.npy')}
 manifest['completed_at']=time.time();save('prepared.json',manifest)
if __name__=='__main__':main()
