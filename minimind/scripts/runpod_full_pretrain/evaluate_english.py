"""Re-load both final pretrain models and run matched English-only probes."""
import gc,json,os,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from common import ROOT,write,sha
OUT=ROOT/'english-tests'
PROMPTS=[
 ('continuation','Learning a new language requires'),
 ('science','Plants need sunlight because'),
 ('science','The human heart'),
 ('fact','Water freezes at'),
 ('fact','A triangle has'),
 ('reasoning','All birds have feathers. A robin is a bird. Therefore, a robin'),
 ('arithmetic','Question: What is 2 + 3?\nAnswer:'),
 ('arithmetic','Question: What is 7 times 8?\nAnswer:'),
 ('story','The little boy opened the door and saw'),
]
def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(4);os.environ['TOKENIZERS_PARALLELISM']='false'
 assert torch.cuda.is_available() and '5090' in torch.cuda.get_device_name()
 tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True);assert len(tok)==6400
 all_models=[]
 for kind in ['dense','moe']:
  report=json.loads((ROOT/kind/'completion.json').read_text());weights=Path(report['weights']);assert sha(weights)==report['weights_sha256']
  config=MiniMindConfig(**json.loads((ROOT/kind/'config.json').read_text()));assert config.use_moe==(kind=='moe')
  model=MiniMindForCausalLM(config).cuda().eval();state=torch.load(weights,map_location='cpu',weights_only=True);assert all(torch.isfinite(v).all() for v in state.values());model.load_state_dict(state,strict=True);del state
  settings=dict(seed=42,max_new_tokens=96,repetition_penalty=1.,max_total_tokens=340,tokenizer_size=6400,dtype='bfloat16',input_format='raw text, no chat template, no implicit BOS, no few-shot examples')
  records=[]
  for mode in ['greedy','sample']:
   for category,prompt in PROMPTS:
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    x=tok(prompt,return_tensors='pt',add_special_tokens=False).input_ids.cuda();assert x.shape[1]+96<=340
    torch.cuda.synchronize();started=time.time()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
     a=model.generate(inputs=x,max_new_tokens=96,do_sample=mode=='sample',temperature=.8 if mode=='sample' else 1.,top_p=.9 if mode=='sample' else 1.,top_k=40 if mode=='sample' else 0,repetition_penalty=1.,eos_token_id=2,use_cache=True)
    torch.cuda.synchronize();new=a[0,x.shape[1]:].tolist()
    if 2 in new:new=new[:new.index(2)+1]
    row=dict(category=category,prompt=prompt,mode=mode,response=tok.decode(new,skip_special_tokens=True),generated_tokens=len(new),eos_reached=bool(new and new[-1]==2),seconds=time.time()-started,temperature=.8 if mode=='sample' else 1.,top_p=.9 if mode=='sample' else 1.,top_k=40 if mode=='sample' else 0)
    records.append(row);write(OUT/(kind+'.json'),dict(model=kind,settings=settings,weights=str(weights),weights_sha256=report['weights_sha256'],strict_reload=True,records=records,complete=False))
    write(OUT/'status.json',dict(phase='evaluating',model=kind,completed=len(records),total=18,timestamp=time.time()))
    print(kind,mode,category,len(records),flush=True)
  result=dict(model=kind,settings=settings,weights=str(weights),weights_sha256=report['weights_sha256'],strict_reload=True,finite_weights=True,records=records,complete=True)
  write(OUT/(kind+'.json'),result);all_models.append(result)
  del model;gc.collect();torch.cuda.empty_cache()
 write(OUT/'results.json',dict(complete=True,models=all_models,total_generations=36,note='English handwritten probes, not a standardized benchmark. Final pretrain checkpoints, not SFT weights.',completed_at=time.time()))
 write(OUT/'status.json',dict(phase='complete',total_generations=36,timestamp=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(OUT/'status.json',dict(phase='failed',type=type(e).__name__,detail=str(e)));raise
