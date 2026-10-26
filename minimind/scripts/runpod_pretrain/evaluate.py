"""Evaluate saved Dense/MoE weights with fixed holdout, factual probes and generation."""
import gc,hashlib,json,math,os,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
ROOT=Path('/shared-volume/minimind-pretrain-rtx5090-20261023')
OUT=ROOT/'tests-20261024'
OUT.mkdir(exist_ok=True)

def save(n,data):
 p=OUT/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2));t.replace(p)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

PROMPTS=[
 ('science','Photosynthesis is the process by which plants'),
 ('science','Water evaporates when'),
 ('science','The human heart is responsible for'),
 ('technology','A computer program is a set of instructions that'),
 ('education','When learning a new language, it is helpful to'),
 ('history','The Industrial Revolution changed society by'),
 ('environment','Recycling helps the environment because'),
 ('reasoning','If all birds have feathers and a robin is a bird, then'),
 ('arithmetic','Question: What is 2 + 3?\nAnswer:'),
 ('arithmetic','Question: What is 7 times 8?\nAnswer:'),
]
PAIRS=[
 ('Water freezes at',' zero degrees Celsius.',' one hundred degrees Celsius.'),
 ('The Earth revolves around',' the Sun.',' the Moon.'),
 ('Plants absorb carbon dioxide and release',' oxygen.',' gold.'),
 ('The organ that pumps blood is the',' heart.',' liver.'),
 ('Humans breathe using their',' lungs.',' kidneys.'),
 ('A triangle has',' three sides.',' four sides.'),
 ('There are seven days in a',' week.',' year.'),
 ('The opposite of hot is',' cold.',' tall.'),
 ('Ice is water in the',' solid state.',' gaseous state.'),
 ('Gravity pulls objects toward',' the Earth.',' the sky.'),
 ('The chemical formula for water is',' H2O.',' CO2.'),
 ('A mammal feeds its young with',' milk.',' sand.'),
]

def main():
 torch.set_num_threads(4);torch.manual_seed(42)
 torch.backends.cuda.matmul.allow_tf32=False
 tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True)
 manifest=json.loads((ROOT/'dataset_manifest.json').read_text())
 targets=set(manifest['validation_indices']);rows={}
 save('status.json',dict(phase='read_holdout'))
 with Path(manifest['source']).joinpath('pretrain-score-3.5.jsonl').open('rb') as f:
  for i,line in enumerate(f):
   if i in targets:rows[i]=json.loads(line)['text']
   if len(rows)==len(targets):break
 assert len(rows)==256
 heldout=[]
 for i in manifest['validation_indices']:
  ids=[tok.bos_token_id]+tok(rows[i],add_special_tokens=False,truncation=True,max_length=338).input_ids+[tok.eos_token_id]
  ids=ids+[tok.pad_token_id]*(340-len(ids));heldout.append(ids)
 heldout=torch.tensor(heldout,dtype=torch.long)
 reports=[]
 for name in ['dense','moe']:
  save('status.json',dict(phase=name+'_evaluation'))
  trained=json.loads((ROOT/name/'completion.json').read_text())
  config=json.loads((ROOT/name/'config.json').read_text())
  weight=Path(trained['weights']);digest=sha(weight)
  assert digest==trained['weights_sha256']
  state=torch.load(weight,map_location='cpu',weights_only=True)
  assert all(torch.isfinite(v).all() for v in state.values())
  model=MiniMindForCausalLM(MiniMindConfig(**config))
  model.load_state_dict(state,strict=True);del state
  model=model.cuda().eval()
  report=dict(model=name,weights=str(weight),sha256=digest,strict_reload=True,finite_weights=True,
              parameters=sum(p.numel() for p in model.parameters()),gpu=torch.cuda.get_device_name(),
              torch=torch.__version__,precision='BF16 autocast with FP32 parameters',max_seq_len=340,
              seed=42,validation_documents=256,generation_settings=dict(max_new_tokens=96,temperature=.8,top_p=.9,top_k=40,repetition_penalty=1.05),
              prompts=PROMPTS)
  nll=tokens=0
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for x in heldout.split(16):
    x=x.cuda();y=x.clone();y[y==tok.pad_token_id]=-100
    result=model(x,labels=y);count=int((y[:,1:]!=-100).sum());value=float(result.loss)
    assert math.isfinite(value)
    nll+=value*count;tokens+=count
  report['validation']=dict(loss=nll/tokens,perplexity=math.exp(nll/tokens),tokens=tokens,
      training_report_loss=trained['validation_after']['loss'],loss_delta=nll/tokens-trained['validation_after']['loss'])
  assert abs(report['validation']['loss_delta'])<.02
  # FP32 cached forward must reproduce the full-prefix final-token logits.
  ids=tok(tok.bos_token+'A computer program is a set of instructions',return_tensors='pt').input_ids.cuda()
  with torch.inference_mode():
   full=model(ids).logits[:,-1].float()
   prefix=model(ids[:,:-1],use_cache=True)
   cached=model(ids[:,-1:],past_key_values=prefix.past_key_values,use_cache=True).logits[:,-1].float()
  error=float((full-cached).abs().max());report['kv_cache']=dict(max_abs_logit_difference=error,passed=error<.001)
  assert report['kv_cache']['passed']
  probes=[]
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for prefix,correct,incorrect in PAIRS:
    scores=[]
    for continuation in [correct,incorrect]:
     a=tok(prefix,add_special_tokens=False).input_ids;b=tok(continuation,add_special_tokens=False).input_ids
     x=torch.tensor([[tok.bos_token_id]+a+b],device='cuda');y=x.clone();y[:,:1+len(a)]=-100
     scores.append(-float(model(x,labels=y).loss))
    probes.append(dict(prefix=prefix,correct=correct,incorrect=incorrect,correct_mean_logprob=scores[0],incorrect_mean_logprob=scores[1],correct_preferred=scores[0]>scores[1]))
  report['factual_probes']=dict(correct=sum(p['correct_preferred'] for p in probes),total=len(probes),items=probes,
      caveat='Handwritten likelihood sanity checks; not a standardized benchmark.')
  generated=[]
  for mode in ['greedy','sample']:
   for category,prompt in PROMPTS:
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    ids=tok(tok.bos_token+prompt,return_tensors='pt').input_ids.cuda()
    assert ids.shape[1]+96<=340
    torch.cuda.synchronize();started=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
     output=model.generate(inputs=ids,max_new_tokens=96,do_sample=mode=='sample',temperature=.8,top_p=.9,top_k=40,repetition_penalty=1.05,use_cache=True,eos_token_id=tok.eos_token_id)
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    continuation=output[0,ids.shape[1]:].tolist();text=tok.decode(continuation,skip_special_tokens=True)
    grams=[tuple(continuation[i:i+3]) for i in range(max(0,len(continuation)-2))]
    result=dict(category=category,prompt=prompt,mode=mode,text=text,generated_tokens=len(continuation),
        eos_reached=bool(continuation and continuation[-1]==tok.eos_token_id),seconds=elapsed,
        tokens_per_second=len(continuation)/elapsed,repeated_trigram_fraction=1-len(set(grams))/max(len(grams),1))
    generated.append(result)
    save(name+'-generations.json',generated)
    print(name,mode,category,repr(text),flush=True)
  report['generations']=generated
  report['nonempty_generations']=sum(bool(g['text'].strip()) for g in generated)
  save(name+'-report.json',report);reports.append(report)
  del model;gc.collect();torch.cuda.empty_cache()
 save('results.json',dict(complete=True,models=reports,source_manifest=manifest,completed_at=time.time()))
 save('status.json',dict(phase='complete'))

if __name__=='__main__':
 try:main()
 except BaseException as e:
  save('status.json',dict(phase='failed',error=type(e).__name__,detail=str(e)));raise
