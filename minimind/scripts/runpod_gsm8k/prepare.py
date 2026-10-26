"""Freeze complete assistant targets with bounded-context overlapping windows."""
import json,shutil,sys,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dataset.lm_dataset import SFTDataset
from common import *
def main():
 ROOT.mkdir(exist_ok=True)
 if (ROOT/'prepared.json').exists():return
 (ROOT/'tokenizer').mkdir(exist_ok=True)
 for source in (BASE/'tokenizer').iterdir():
  if source.is_file():shutil.copyfile(source,ROOT/'tokenizer'/source.name)
 tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True);assert len(tok)==6400 and tok.pad_token_id==0
 native=SFTDataset.__new__(SFTDataset);native.tokenizer=tok
 native.bos_id=tok(tok.bos_token+'assistant\n',add_special_tokens=False).input_ids
 native.eos_id=tok(tok.eos_token+'\n',add_special_tokens=False).input_ids
 manifest=dict(dataset='openai/gsm8k',config='main',revision=REV,max_seq_len=LENGTH,epochs=2,tokenizer_size=len(tok),template='original tokenizer chat template, open_thinking=False, no random augmentation',loss_mask='native SFTDataset.generate_labels; assistant only',window_policy='340 length, stride256, overlap84, each assistant target exactly once; all original rationale/final-answer tokens retained',files=[],splits={})
 questions={}
 for split,expected in [('train',7473),('test',1319)]:
  raw=Path(hf_hub_download('openai/gsm8k',repo_type='dataset',filename=f'main/{split}-00000-of-00001.parquet',revision=REV,cache_dir='/tmp/gsm8k-hf'))
  dest=ROOT/'raw'/raw.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(raw,dest)
  rows=pq.read_table(raw).to_pylist();assert len(rows)==expected
  manifest['files'].append(dict(split=split,path=str(dest),sha256=sha(dest)))
  xs=[];ys=[];records=[];long=target_tokens=0;questions[split]=set()
  for i,row in enumerate(rows):
   q=row['question'];a=row['answer'];assert q not in questions[split];questions[split].add(q)
   assert extract(a) is not None
   chat=tok.apply_chat_template(messages(q,a),tokenize=False,add_generation_prompt=False,open_thinking=False)
   ids=tok(chat,add_special_tokens=False).input_ids;native.max_length=len(ids);labels=native.generate_labels(ids)
   pairs=windows(ids,labels);assert pairs
   long+=len(ids)>LENGTH;target_tokens+=sum(v!=-100 for v in labels[1:])
   xs.extend(x for x,y in pairs);ys.extend(y for x,y in pairs)
   records.append(dict(index=i,question=q,answer=a,gold=extract(a),windows=len(pairs),tokens=len(ids)))
  for name,data in [('input',xs),('labels',ys)]:np.save(ROOT/f'{split}-{name}.npy',np.array(data,dtype=np.int32))
  write(ROOT/f'{split}-records.json',records)
  manifest['splits'][split]=dict(rows=len(rows),windows=len(xs),conversations_over_340=long,assistant_target_tokens=target_tokens,discarded_rows=0,discarded_target_tokens=0)
 assert not (questions['train'] & questions['test'])
 manifest['arrays']={p.name:sha(p) for p in ROOT.glob('*.npy')}
 manifest['test_records_sha256']=sha(ROOT/'test-records.json');manifest['completed_at']=time.time()
 write(ROOT/'prepared.json',manifest);print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
