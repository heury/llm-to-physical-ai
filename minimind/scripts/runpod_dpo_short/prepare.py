"""Select exact untruncated <340-token preference pairs using the base tokenizer."""
import collections,hashlib,json,unicodedata
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
ROOT=Path('outputs/minimind/dpo-short-20261025')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())
def render(tok,msgs,generation=False):
 return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=generation,open_thinking=False).replace('<think>\n\n</think>\n\n','')
def main():
 tok=AutoTokenizer.from_pretrained(ROOT/'base',local_files_only=True);tok.model_max_length=2**30;assert len(tok)==6400
 source=Path((ROOT/'source-path.txt').read_text());eligible=[];counts=collections.Counter();total=0
 for batch in pq.ParquetFile(source).iter_batches(batch_size=128):
  valid=[]
  for row in batch.to_pylist():
   idx=total;total+=1
   c,r=row['chosen'],row['rejected']
   if not c or not r or c[-1]['role']!='assistant' or r[-1]['role']!='assistant' or c[:-1]!=r[:-1]:counts['invalid_or_mismatched_prompt']+=1;continue
   if not c[-1]['content'].strip() or not r[-1]['content'].strip():counts['empty_response']+=1;continue
   prefix=render(tok,c[:-1],True);ct=render(tok,c);rt=render(tok,r)
   if not ct.startswith(prefix) or not rt.startswith(prefix):counts['invalid_template_prefix']+=1;continue
   valid.append((idx,row,prefix,ct,rt))
  flat=[s for _,_,p,c,r in valid for s in [p,c,r]]
  if not flat:continue
  enc=tok(flat,add_special_tokens=False,truncation=False,return_attention_mask=False).input_ids
  for j,(idx,row,p,c,r) in enumerate(valid):
   prefix,chosen,rejected=enc[j*3:j*3+3];n=len(prefix)
   if max(len(chosen),len(rejected))>=340:counts['length_ge_340']+=1;continue
   if chosen[:n]!=prefix or rejected[:n]!=prefix:counts['token_prefix_mismatch']+=1;continue
   if len(chosen)<=n or len(rejected)<=n:counts['no_target']+=1;continue
   if chosen==rejected:counts['identical_tokens']+=1;continue
   assert tok.eos_token_id in chosen[n:] and tok.eos_token_id in rejected[n:]
   prompt_key=hashlib.sha256(canonical(json.dumps(row['chosen'][:-1],ensure_ascii=False,sort_keys=True)).encode()).hexdigest()
   record=dict(row,source_row=idx,prompt_sha256=prompt_key,prompt_tokens=n,chosen_tokens=len(chosen),rejected_tokens=len(rejected),chosen_response_tokens=len(chosen)-n,rejected_response_tokens=len(rejected)-n)
   eligible.append((hashlib.sha256(f'dpo-short42:{idx}'.encode()).hexdigest(),record,chosen,rejected));counts['length_eligible']+=1
  if total%4096==0:print(json.dumps({'scanned':total,'eligible':len(eligible),'counts':dict(counts)}),flush=True)
 chosen_rows=[];seen=set()
 for entry in sorted(eligible):
  key=entry[1]['prompt_sha256']
  if key in seen:counts['duplicate_eligible_prompt']+=1;continue
  seen.add(key);chosen_rows.append(entry)
 assert len(chosen_rows)>=3000, f'Only {len(chosen_rows)} unique eligible prompts, no truncation permitted'
 selected=chosen_rows[:3000];data=ROOT/'data';data.mkdir(exist_ok=True);details={};splitkeys=[]
 for split,start,end in [('train',0,2700),('validation',2700,2850),('test',2850,3000)]:
  subset=selected[start:end];rows=[];arr=np.zeros((len(subset),2,2,340),dtype=np.int16);arr[:,:,1,:]=-100
  for i,(_,row,c,r) in enumerate(subset):
   rows.append(row)
   for side,ids in enumerate([c,r]):
    arr[i,side,0,:len(ids)]=ids;arr[i,side,1,row['prompt_tokens']:len(ids)]=ids[row['prompt_tokens']:]
    assert not ((arr[i,side,1]!=-100)&(arr[i,side,0]==0)).any()
  np.save(data/f'{split}.npy',arr);pq.write_table(pa.Table.from_pylist(rows),data/f'{split}.parquet',compression='zstd')
  (data/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
  details[split]={'pairs':len(rows),'max_chosen_tokens':max(r['chosen_tokens'] for r in rows),'max_rejected_tokens':max(r['rejected_tokens'] for r in rows),'rating_ties':sum(r['chosen-rating']==r['rejected-rating'] for r in rows),'sources':dict(collections.Counter(r['source'] for r in rows))};splitkeys.append({r['prompt_sha256'] for r in rows})
 assert not(splitkeys[0]&splitkeys[1] or splitkeys[0]&splitkeys[2] or splitkeys[1]&splitkeys[2])
 manifest={'source':json.loads((ROOT/'dataset-source.json').read_text()),'base':json.loads((ROOT/'model-source.json').read_text()),'source_parquet_sha256':sha(source),'source_rows':total,'filter_counts':dict(counts),'unique_eligible_prompts':len(chosen_rows),'selected_pairs':3000,'seed':42,'splits':details,'length_rule':'Both full rendered prompt+response sequences, including chat special tokens and EOS, strictly <340 tokens; no truncation; pad to340 only after selection','template':'Original base tokenizer, open_thinking=False, deterministic removal of empty think block; no extra system prompt beyond original template','target_mask':'Final assistant completion including EOS/newline; prompt and padding masked; sum token log probabilities','selection':'Hash-random ordered unique canonical prompts; 2700 train /150 validation /150 test; original preference labels and ties preserved','decontamination':'DPO prompt groups disjoint; prior SFT/pretraining exposure not excluded','files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(data.iterdir())}}
 (ROOT/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
