"""Zero-shot GSM8K generation, answer extraction and assistant-only PPL."""
import collections,json,math,time
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from common import *
PROBES=[('What is 2 + 3? Give only the answer.','5'),('What is 7 times 8? Give only the answer.','56'),('What is 12 minus 5? Give only the answer.','7'),('What is 20 divided by 4? Give only the answer.','5'),('I have 3 apples and buy 4 more. How many apples do I have?','7'),('A book costs 6 dollars. How much do 3 books cost?','18')]
class Arrays(Dataset):
 def __init__(self,split):
  self.x=np.load(RUNTIME/f'{split}-input.npy',mmap_mode='r');self.y=np.load(RUNTIME/f'{split}-labels.npy',mmap_mode='r')
 def __len__(self):return len(self.x)
 def __getitem__(self,i):return torch.tensor(self.x[i],dtype=torch.long),torch.tensor(self.y[i],dtype=torch.long)
@torch.inference_mode()
def validation(model):
 model.eval();nll=targets=0
 with torch.autocast('cuda',dtype=torch.bfloat16):
  for x,y in DataLoader(Arrays('test'),batch_size=32,num_workers=0):
   x=x.cuda();y=y.cuda();n=int((y[:,1:]!=-100).sum());loss=float(model(x,labels=y).loss);assert math.isfinite(loss)
   nll+=n*loss;targets+=n
 return dict(loss=nll/targets,perplexity=math.exp(nll/targets),assistant_target_tokens=targets)
@torch.inference_mode()
def evaluate(model,tokenizer,out,label,status):
 model.eval();started=time.time();records=json.loads((ROOT/'test-records.json').read_text());groups=collections.defaultdict(list);results=[]
 for row in records:
  text=tokenizer.apply_chat_template(messages(row['question']),tokenize=False,add_generation_prompt=True,open_thinking=False)
  ids=tokenizer(text,add_special_tokens=False).input_ids
  if len(ids)>=LENGTH:
   results.append(dict(index=row['index'],question=row['question'],gold=row['gold'],response='',strict_prediction=None,flexible_prediction=None,strict_correct=False,flexible_correct=False,prompt_tokens=len(ids),unevaluable_context=True));continue
  # No truncation or gold answer in prompts; exact-length batches avoid left-padding changes.
  groups[len(ids)].append((row,ids))
 generated_batches=0
 for length,group in sorted(groups.items()):
  for start in range(0,len(group),32):
   chunk=group[start:start+32];x=torch.tensor([ids for row,ids in chunk],device='cuda');budget=min(256,LENGTH-length)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    answer=model.generate(inputs=x,max_new_tokens=budget,do_sample=False,temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,use_cache=True,eos_token_id=tokenizer.eos_token_id)
   for (row,ids),output in zip(chunk,answer[:,length:].tolist()):
    ended=tokenizer.eos_token_id in output
    if ended:output=output[:output.index(tokenizer.eos_token_id)+1]
    text=tokenizer.decode(output,skip_special_tokens=True);strict=extract(text);flex=extract(text,False)
    results.append(dict(index=row['index'],question=row['question'],gold=row['gold'],response=text,strict_prediction=strict,flexible_prediction=flex,strict_correct=strict==row['gold'],flexible_correct=flex==row['gold'],prompt_tokens=length,generated_tokens=len(output),eos_reached=ended,token_budget=budget,unevaluable_context=False))
   generated_batches+=1
   if generated_batches%5==0:
    write(out/f'generations_{label}.json',sorted(results,key=lambda r:r['index']));status('tests_'+label,evaluated=len(results),total=len(records),seconds=time.time()-started)
 assert len(results)==1319 and len({r['index'] for r in results})==1319
 results.sort(key=lambda r:r['index']);write(out/f'generations_{label}.json',results)
 probes=[]
 for question,gold in PROBES:
  text=tokenizer.apply_chat_template([{'role':'user','content':question}],tokenize=False,add_generation_prompt=True,open_thinking=False)
  x=tokenizer(text,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   a=model.generate(inputs=x,max_new_tokens=128,do_sample=False,temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,use_cache=True,eos_token_id=2)
  text=tokenizer.decode(a[0,x.shape[1]:],skip_special_tokens=True)
  probes.append(dict(prompt=question,gold=gold,response=text,prediction=extract(text,False),correct=extract(text,False)==number(gold)))
 strict=sum(r['strict_correct'] for r in results);flex=sum(r['flexible_correct'] for r in results)
 report=dict(phase=label,dataset='openai/gsm8k',config='main',revision=REV,test_rows=1319,strict_correct=strict,strict_accuracy=strict/1319,flexible_correct=flex,flexible_accuracy=flex/1319,valid_final_format=sum(r['strict_prediction'] is not None for r in results),context_overflow=sum(r['unevaluable_context'] for r in results),token_budget_exhausted=sum(not r.get('eos_reached',False) for r in results if not r['unevaluable_context']),validation=validation(model),probes=probes,seconds=time.time()-started,generation_settings=dict(few_shot_examples=0,do_sample=False,repetition_penalty=1.,max_new_tokens=256,total_context_limit=LENGTH,batch_size_max=32,padding='none; batch by exact prompt-token length',prompt_suffix=SUFFIX),scoring='strict numeric final line #### number; secondary fallback last numeric token. All 1319 in denominator, including context overflow/invalid output.')
 write(out/f'tests_{label}.json',report);model.train();return report
