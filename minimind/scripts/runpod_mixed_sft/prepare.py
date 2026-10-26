"""Pin official splits, exclude heldout prompt overlap, dedup, preserve all targets."""
import collections,json,os,shutil,sys,time
from pathlib import Path
import numpy as np
import ahocorasick
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dataset.lm_dataset import SFTDataset
from common import *
REPOS={'ultrachat':'HuggingFaceH4/ultrachat_200k','smoltalk':'HuggingFaceTB/smoltalk','gsm8k':'openai/gsm8k'}
FILES={
 'gsm8k':{'train':['main/train-00000-of-00001.parquet'],'test':['main/test-00000-of-00001.parquet']},
 'ultrachat':{'train':[f'data/train_sft-{n:05d}-of-00003-{suffix}.parquet' for n,suffix in enumerate(['a3ecf92756993583','0a1804bcb6ae68c6','ee46ed25cfae92c6'])],'test':['data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet']},
 'smoltalk':{'train':[f'data/all/train-{n:05d}-of-00009.parquet' for n in range(9)],'test':['data/all/test-00000-of-00001.parquet']}}
def main():
 ROOT.mkdir(parents=True,exist_ok=True);RUNTIME.mkdir(exist_ok=True)
 if (ROOT/'prepared.json').exists():return
 (ROOT/'tokenizer').mkdir(exist_ok=True)
 for p in (BASE/'tokenizer').iterdir():
  if p.is_file():shutil.copyfile(p,ROOT/'tokenizer'/p.name)
 tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True);assert len(tok)==6400 and tok.pad_token_id==0;tok.model_max_length=2**30
 bos=tok(tok.bos_token+'assistant\n',add_special_tokens=False).input_ids;eos=tok(tok.eos_token+'\n',add_special_tokens=False).input_ids
 native=SFTDataset.__new__(SFTDataset);native.tokenizer=tok;native.bos_id=bos;native.eos_id=eos
 manifest=dict(datasets={k:dict(repo=REPOS[k],revision=REVS[k],config='all' if k=='smoltalk' else 'main' if k=='gsm8k' else 'train_sft/test_sft') for k in REPOS},max_seq_len=340,stride=256,overlap=84,tokenizer_size=6400,epochs=2,mixing='Natural retained training conversation/window counts; uniform shuffle of all windows, no dataset oversampling',template='Original tokenizer template with deterministic removal of empty think blocks for both training and inference',policy='Full conversations, assistant-only labels, 340 windows stride256, overlap labels masked after first use; final short window padded, never drop assistant/EOS targets',files=[],splits={},filters='Canonical first-user prompt overlap against every official test split excluded from training; GSM8K raw and suffixed prompt forms blocked. Exact normalized full-conversation duplicates removed. Normalized GSM8K test-question substrings in any message also excluded. No guarantee of paraphrase-level decontamination.')
 paths={s:{} for s in REPOS};started=time.time()
 # Download tests before training data; prepare heldout prompt protection first.
 for split in ['test','train']:
  for source in ['gsm8k','ultrachat','smoltalk']:
   found=[]
   for filename in FILES[source][split]:
    write(ROOT/'prepare_status.json',dict(phase='download',source=source,split=split,file=filename,timestamp=time.time()))
    cache=Path(hf_hub_download(REPOS[source],repo_type='dataset',filename=filename,revision=REVS[source],cache_dir=str(RUNTIME/'hf')))
    dest=ROOT/'raw'/source/Path(filename).name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(cache,dest)
    digest=sha(cache);assert sha(dest)==digest;manifest['files'].append(dict(source=source,split=split,file=filename,sha256=digest,bytes=cache.stat().st_size));found.append(cache)
   paths[source][split]=found
 handles={name:(RUNTIME/(name+'.bin')).open('wb') for name in ['train','validation_gsm8k','validation_ultrachat','validation_smoltalk']}
 stats={name:collections.Counter() for name in handles};blocked=set();seen=set();train_prompts=set();component_counts=collections.Counter();audit=(RUNTIME/'audit.jsonl').open('w');gsm_records=[]
 def batches(source,split):
  for path in paths[source][split]:
   for batch in pq.ParquetFile(path).iter_batches(batch_size=128):yield batch.to_pylist()
 def encode(rows,source,name,indices):
  valid=[]
  for idx,row,msg in rows:valid.append((idx,row,msg,chat_text(tok,msg)))
  if not valid:return
  tokens=tok([v[3] for v in valid],add_special_tokens=False,truncation=False,return_attention_mask=False,return_token_type_ids=False).input_ids
  for (idx,row,msg,_),ids in zip(valid,tokens):
   spans=assistant_spans(ids,bos,eos);a=np.asarray(ids,dtype=np.int16);labels=np.full(len(ids),-100,dtype=np.int16)
   for begin,end in spans:labels[begin:end]=a[begin:end]
   if stats[name]['native_mask_checks']<128:
    native.max_length=len(ids);assert labels.tolist()==native.generate_labels(ids);stats[name]['native_mask_checks']+=1
   expected=int((labels[1:]!=-100).sum());expected_eos=int(((labels==tok.eos_token_id)&(labels!=-100)).sum())
   if not expected:stats[name]['no_targets']+=1;continue
   covered=1;written=written_eos=windows=0;first=stats[name]['windows']
   for start,end in window_bounds(len(ids)):
    y=labels[start:end].copy();y[:max(1,covered-start)]=-100;covered=max(covered,end)
    active=y[1:]!=-100
    if not active.any():stats[name]['no_target_windows']+=1;continue
    pair=np.empty((2,LENGTH),dtype=np.int16);pair[0].fill(0);pair[1].fill(-100);pair[0,:end-start]=a[start:end];pair[1,:end-start]=y
    handles[name].write(pair.tobytes());written+=int(active.sum());written_eos+=int((y[1:]==tok.eos_token_id).sum());windows+=1
   assert written==expected and written_eos==expected_eos and windows>0
   st=stats[name];st['conversations']+=1;st['windows']+=windows;st['assistant_targets']+=written;st['assistant_eos_targets']+=written_eos;st['conversations_over_340']+=len(ids)>LENGTH
   if name=='train':st['source_'+source+'_rows']+=1;st['source_'+source+'_windows']+=windows;st['source_'+source+'_targets']+=written
   audit.write(json.dumps(dict(dataset=source,split=name,index=idx,first_window=first,windows=windows,assistant_targets=written,eos_targets=written_eos))+'\n')
 for source in ['gsm8k','ultrachat','smoltalk']:
  total=sum(pq.ParquetFile(p).metadata.num_rows for p in paths[source]['test']);keep=set(range(total)) if source=='gsm8k' else set(np.random.default_rng(42).choice(total,size=min(512,total),replace=False).tolist())
  counter=0
  for chunk in batches(source,'test'):
   selected=[]
   for row in chunk:
    idx=counter;counter+=1;msg=normalize(row,source)
    if not msg:continue
    blocked.add(prompt_key(msg))
    if source=='gsm8k':
     blocked.add(prompt_key([{'role':'user','content':row['question']}]))
     gold=extract(row['answer']);assert gold is not None;gsm_records.append(dict(index=idx,question=row['question'],answer=row['answer'],gold=gold))
    if idx in keep:selected.append((idx,row,msg))
   encode(selected,source,'validation_'+source,None)
  assert counter==total;manifest['splits'][source]=dict(official_test_rows=total,validation_conversations_sampled=len(keep))
 assert len(gsm_records)==1319;write(ROOT/'test-records.json',gsm_records)
 gsm_guard=ahocorasick.Automaton()
 for rec in gsm_records:gsm_guard.add_word(canonical(rec['question']),rec['index'])
 gsm_guard.make_automaton()
 # GSM8K first preserves its original requested examples under exact-dialogue dedup.
 for source in ['gsm8k','ultrachat','smoltalk']:
  counter=0;filtered=collections.Counter()
  for chunk in batches(source,'train'):
   selected=[]
   for row in chunk:
    idx=counter;counter+=1;msg=normalize(row,source)
    if not msg:filtered['unsupported_or_missing_messages']+=1;continue
    key=prompt_key(msg)
    if key in blocked or (source=='gsm8k' and prompt_key([{'role':'user','content':row['question']}]) in blocked):filtered['official_test_prompt_overlap']+=1;continue
    if any(next(gsm_guard.iter(canonical(m['content'])),None) is not None for m in msg):filtered['gsm8k_test_question_substring_overlap']+=1;continue
    ck=conversation_key(msg)
    if ck in seen:filtered['duplicate_conversation']+=1;continue
    seen.add(ck);train_prompts.add(key);selected.append((idx,row,msg))
    if source=='smoltalk':component_counts[str(row.get('source','unknown'))]+=1
   encode(selected,source,'train',None)
   if counter//4096 != (counter-len(chunk))//4096:
    progress=dict(phase='tokenizing',source=source,source_rows_seen=counter,stats={k:dict(v) for k,v in stats.items()},seconds=time.time()-started,timestamp=time.time());write(ROOT/'prepare_status.json',progress);print(json.dumps(progress),flush=True)
  manifest['splits'][source].update(official_train_rows=counter,filters=dict(filtered))
 assert not (train_prompts & blocked)
 data=ROOT/'data';data.mkdir(exist_ok=True);manifest['arrays']={};manifest['validation_indices']={}
 for name,handle in handles.items():
  handle.flush();os.fsync(handle.fileno());handle.close();count=stats[name]['windows'];assert count>0
  local=RUNTIME/(name+'.bin');assert local.stat().st_size==count*2*LENGTH*2
  write(ROOT/'prepare_status.json',dict(phase='publish_arrays',file=name,windows=count,timestamp=time.time()))
  dest=data/(name+'.bin');shutil.copyfile(local,dest);digest=sha(local);assert sha(dest)==digest
  manifest['arrays'][name]=dict(path=str(dest),sha256=digest,bytes=dest.stat().st_size,rows=count,dtype='int16',shape=[count,2,LENGTH])
  if name!='train':manifest['validation_indices'][name]=np.random.default_rng(42).choice(count,size=min(512,count),replace=False).tolist()
 audit.close();shutil.copyfile(RUNTIME/'audit.jsonl',data/'audit.jsonl')
 manifest.update(complete=True,stats={k:dict(v) for k,v in stats.items()},smoltalk_components=dict(component_counts),prompt_split_disjoint=True,test_records_sha256=sha(ROOT/'test-records.json'),preparation_seconds=time.time()-started,completed_at=time.time())
 write(ROOT/'prepared.json',manifest);write(ROOT/'prepare_status.json',dict(phase='complete',completed_at=time.time(),train_windows=stats['train']['windows']));print('PREPARED',json.dumps(manifest),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(ROOT/'prepare_status.json',dict(phase='failed',type=type(e).__name__,detail=str(e),timestamp=time.time()));raise
