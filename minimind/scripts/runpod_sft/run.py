"""One independent RTX5090 full-SFT job, before/after tests and persistent export."""
import argparse,gc,hashlib,json,math,os,re,shutil,subprocess,sys,tarfile,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset,Subset
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from trainer.trainer_utils import setup_seed,SkipBatchSampler,lm_checkpoint
import trainer.train_full_sft as native
ROOT=Path('/shared-volume/minimind-ultrachat-sft-20261024')
BASE=Path('/shared-volume/minimind-pretrain-rtx5090-20261023')
RUNTIME=Path('/tmp/minimind-sft')
PROMPTS=[
 ('science','Explain photosynthesis in two short sentences.'),
 ('science','Why does water evaporate? Answer in one sentence.'),
 ('science','What does the human heart do?'),
 ('technology','Explain what a computer program is to a beginner.'),
 ('instruction','Give exactly three tips for learning English.'),
 ('instruction','Reply with only the word: apple'),
 ('extraction','Extract the city name from this sentence: Alice lives in Paris.'),
 ('arithmetic','What is 2 + 3? Give only the answer.'),
 ('arithmetic','What is 7 times 8? Give only the answer.'),
 ('reasoning','All birds have feathers. A robin is a bird. Does a robin have feathers?'),
 ('writing','Write a short friendly email thanking a teacher for their help.'),
 ('summary','Summarize in one sentence: Bees pollinate flowers. This helps plants reproduce and supports food production.'),
]
FACTS=[('Water freezes at',' zero degrees Celsius.',' one hundred degrees Celsius.'),('The Earth revolves around',' the Sun.',' the Moon.'),('Plants absorb carbon dioxide and release',' oxygen.',' gold.'),('The organ that pumps blood is the',' heart.',' liver.'),('Humans breathe using their',' lungs.',' kidneys.'),('A triangle has',' three sides.',' four sides.'),('There are seven days in a',' week.',' year.'),('The opposite of hot is',' cold.',' tall.'),('Ice is water in the',' solid state.',' gaseous state.'),('Gravity pulls objects toward',' the Earth.',' the sky.'),('The chemical formula for water is',' H2O.',' CO2.'),('A mammal feeds its young with',' milk.',' sand.')]

def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False));t.replace(p)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
class FrozenSFT(Dataset):
 def __init__(self,split):
  self.x=np.load(RUNTIME/(split+'-input.npy'),mmap_mode='r');self.y=np.load(RUNTIME/(split+'-labels.npy'),mmap_mode='r')
 def __len__(self):return len(self.x)
 def __getitem__(self,i):return torch.tensor(self.x[i],dtype=torch.long),torch.tensor(self.y[i],dtype=torch.long)

def main(kind):
 out=ROOT/kind;out.mkdir(parents=True,exist_ok=True)
 for n in ['out','checkpoints','wandb']:(out/n).mkdir(exist_ok=True)
 def phase(n,**kw):write(out/'status.json',dict(phase=n,timestamp=time.time(),**kw));print('PHASE',n,flush=True)
 assert not (out/'completion.json').exists()
 os.environ.update(WANDB_MODE='offline',WANDB_DIR=str(out/'wandb'),WANDB_CACHE_DIR=str(RUNTIME/'wandb-cache'),HF_HOME=str(RUNTIME/'hf'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
 import wandb
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
 gpu=torch.cuda.get_device_properties(0);assert '5090' in gpu.name and torch.cuda.is_bf16_supported()
 torch.cuda.set_per_process_memory_fraction(.94)
 phase('waiting_for_dataset')
 while not (ROOT/'prepared.json').exists():
  if (ROOT/'prepare.exit').exists() and (ROOT/'prepare.exit').read_text().strip()!='0':raise RuntimeError('Dataset preparation failed')
  time.sleep(10)
 manifest=json.loads((ROOT/'prepared.json').read_text())
 for name,digest in manifest['arrays'].items():
  dest=RUNTIME/name
  if not dest.exists():shutil.copyfile(ROOT/name,dest)
  assert sha(dest)==digest
 train=FrozenSFT('train');test=FrozenSFT('test');validation=Subset(test,manifest['validation_indices'])
 tokenizer=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True)
 assert len(tokenizer)==6400
 moe=kind=='moe';suffix='_moe' if moe else ''
 base_report=json.loads((BASE/kind/'completion.json').read_text());base_weights=Path(base_report['weights'])
 assert sha(base_weights)==base_report['weights_sha256']
 config=MiniMindConfig(**json.loads((BASE/kind/'config.json').read_text()))
 state=torch.load(base_weights,map_location='cpu',weights_only=True)
 config.to_json_file(str(out/'config.json'))
 phase('benchmark')
 setup_seed(42);bench=MiniMindForCausalLM(config).cuda().train();bench.load_state_dict(state,strict=True)
 opt=torch.optim.AdamW(bench.parameters(),lr=5e-5);candidates=[]
 for batch in [32,64,128]:
  x=y=res=loss=None;opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  try:
   pairs=[train[i] for i in range(batch)];x=torch.stack([a for a,b in pairs]).cuda();y=torch.stack([b for a,b in pairs]).cuda()
   started=time.time()
   for repeat in range(2):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):res=bench(x,labels=y);loss=res.loss+res.aux_loss
    assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(bench.parameters(),1.,error_if_nonfinite=True)
    if moe:
     routers=[p.grad for n,p in bench.named_parameters() if n.endswith('mlp.gate.weight')]
     assert len(routers)==8 and all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in routers);del routers
    opt.step()
   torch.cuda.synchronize();row=dict(batch_size=batch,peak_bytes=torch.cuda.max_memory_allocated(),seconds_per_batch=(time.time()-started)/2,loss=float(loss),grad_norm=float(norm));candidates.append(row);print('BENCHMARK',json.dumps(row),flush=True)
   if row['peak_bytes']>gpu.total_memory*.85:break
  except torch.OutOfMemoryError:print('BENCHMARK OOM',batch,flush=True);break
  finally:
   opt.zero_grad(set_to_none=True);del x,y,res,loss;gc.collect();torch.cuda.empty_cache()
 safe=[r for r in candidates if r['peak_bytes']<gpu.total_memory*.85];assert safe
 chosen=max(safe,key=lambda r:r['batch_size']/r['seconds_per_batch']);batch=chosen['batch_size'];accum=128//batch
 del bench,opt;gc.collect();torch.cuda.empty_cache();setup_seed(42)
 model=MiniMindForCausalLM(config).cuda();model.load_state_dict(state,strict=True);del state
 optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5)
 steps=math.ceil(len(train)/batch);epochs=2
 settings=dict(model=kind,epochs=epochs,batch_size=batch,accumulation_steps=accum,effective_batch_size=128,learning_rate=5e-5,warmup_microsteps=min(1000,max(1,int(steps*epochs*.03))),max_seq_len=340,dtype='bfloat16',grad_clip=1.,log_interval=10,save_interval=250,num_workers=4,seed=42,steps_per_epoch=steps,optimizer_updates=epochs*math.ceil(steps/accum),training_rows=len(train),validation_rows=len(validation),base_weights=str(base_weights),base_sha256=base_report['weights_sha256'],dataset_revision=manifest['revision'],model_parameters=sum(p.numel() for p in model.parameters()),benchmark=candidates,gpu=gpu.name,torch=torch.__version__)
 write(out/'training_settings.json',settings)
 args=SimpleNamespace(**{k:settings[k] for k in ['epochs','batch_size','accumulation_steps','learning_rate','max_seq_len','dtype','grad_clip','log_interval','save_interval','num_workers']},device='cuda:0',save_dir=str(out/'out'),save_weight='full_sft')
 native.args=args;native.model=model;native.optimizer=optimizer;native.lm_config=config
 native.scaler=torch.amp.GradScaler('cuda',enabled=False);native.autocast_ctx=torch.autocast('cuda',dtype=torch.bfloat16)
 def val():
  model.eval();nll=tokens=0
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for x,y in DataLoader(validation,batch_size=16,num_workers=2):
    x=x.cuda();y=y.cuda();n=int((y[:,1:]!=-100).sum());loss=float(model(x,labels=y).loss);assert math.isfinite(loss)
    nll+=loss*n;tokens+=n
  model.train();return dict(loss=nll/tokens,perplexity=math.exp(nll/tokens),tokens=tokens,documents=len(validation))
 def testing(label):
  phase('tests_'+label);model.eval();records=[]
  for mode in ['greedy','sample']:
   for category,prompt in PROMPTS:
    setup_seed(42)
    text=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,open_thinking=False)
    ids=tokenizer(text,return_tensors='pt',add_special_tokens=False).input_ids.cuda();assert ids.shape[1]+128<=340
    torch.cuda.synchronize();started=time.time()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
     answer=model.generate(inputs=ids,max_new_tokens=128,do_sample=mode=='sample',temperature=.8,top_p=.9,top_k=40,repetition_penalty=1.05,use_cache=True,eos_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize();new=answer[0,ids.shape[1]:].tolist();response=tokenizer.decode(new,skip_special_tokens=True)
    records.append(dict(category=category,prompt=prompt,mode=mode,response=response,generated_tokens=len(new),eos_reached=bool(new and new[-1]==tokenizer.eos_token_id),seconds=time.time()-started,seed=42))
    write(out/('generations_'+label+'.json'),records)
  facts=[]
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for prefix,correct,incorrect in FACTS:
    scores=[]
    for c in [correct,incorrect]:
     a=tokenizer(prefix,add_special_tokens=False).input_ids;b=tokenizer(c,add_special_tokens=False).input_ids
     x=torch.tensor([[tokenizer.bos_token_id]+a+b],device='cuda');y=x.clone();y[:,:len(a)+1]=-100;scores.append(-float(model(x,labels=y).loss))
    facts.append(dict(prefix=prefix,correct=correct,incorrect=incorrect,correct_preferred=scores[0]>scores[1],mean_logprobs=scores))
  result=dict(phase=label,validation=val(),generations=records,factual_probes=dict(correct=sum(f['correct_preferred'] for f in facts),total=len(facts),items=facts),generation_settings=dict(max_new_tokens=128,temperature=.8,top_p=.9,top_k=40,repetition_penalty=1.05,seed=42),note='English-only handwritten probes; not a standardized benchmark. Assistant-only test_sft validation loss.')
  write(out/('tests_'+label+'.json'),result);model.train();return result
 before=testing('before')
 run=wandb.init(project='MiniMind-UltraChat-SFT',name='RTX5090-'+kind+'-2epoch-seq340-20261024',config=settings,dir=str(out/'wandb'),mode='offline')
 write(out/'wandb_run.json',dict(id=run.id,url=f'https://wandb.ai/mind4u/MiniMind-UltraChat-SFT/runs/{run.id}',mode='offline_pending_local_sync'))
 updates=0
 weights=out/f'out/full_sft_768{suffix}.pth'
 def checkpoint(*a,**kw):
  kw['save_dir']=str(out/'checkpoints');lm_checkpoint(*a,**kw)
  write(out/'checkpoint.json',dict(epoch=kw.get('epoch',0)+1,step=kw.get('step'),optimizer_updates=updates,wandb_id=run.id))
 native.lm_checkpoint=checkpoint
 def post_step(opt,argv,kwargs):
  nonlocal updates
  updates+=1
  if updates==1:
   checkpoint(config,weight='full_sft',model=model,optimizer=optimizer,scaler=native.scaler,epoch=0,step=accum,wandb=run)
   shutil.copyfile(out/f'checkpoints/full_sft_768{suffix}.pth',weights)
 optimizer.register_step_post_hook(post_step)
 def lr(step,total,peak):
  warm=settings['warmup_microsteps'];return peak*step/warm if step<=warm else peak*(.1+.45*(1+math.cos(math.pi*(step-warm)/max(total-warm,1))))
 native.get_lr=lr
 def logger(msg):
  print(msg,flush=True)
  m=re.search(r'\((\d+)/(\d+)\), loss: ([\d.e+-]+).*epoch_time: ([\d.]+)',msg)
  if m:
   row=dict(epoch=epoch+1,epochs=epochs,step=int(m[1]),steps_per_epoch=int(m[2]),global_step=epoch*steps+int(m[1]),loss=float(m[3]),eta_minutes_this_epoch=float(m[4]),optimizer_updates=updates,timestamp=time.time());assert math.isfinite(row['loss']);write(out/'progress.json',row)
   with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 native.Logger=logger
 phase('training',settings=settings);started=time.time()
 run.log({'validation/before_loss':before['validation']['loss']})
 for epoch in range(epochs):
  setup_seed(42+epoch);indices=torch.randperm(len(train)).tolist()
  loader=DataLoader(train,batch_sampler=SkipBatchSampler(indices,batch,0),num_workers=4,pin_memory=True)
  native.train_epoch(epoch,loader,steps,wandb=run)
  shutil.copyfile(weights,out/f'out/full_sft_768{suffix}_epoch{epoch+1}.pth')
  evaluation=val();write(out/f'epoch{epoch+1}.json',evaluation);run.log({f'validation/epoch{epoch+1}_loss':evaluation['loss']})
 assert updates==settings['optimizer_updates']
 phase('verify_checkpoint')
 resume_path=out/f'checkpoints/full_sft_768{suffix}_resume.pth'
 saved=torch.load(weights,map_location='cpu',weights_only=True);assert all(torch.isfinite(v).all() for v in saved.values())
 fresh=MiniMindForCausalLM(config);fresh.load_state_dict(saved,strict=True)
 resume=torch.load(resume_path,map_location='cpu',weights_only=False)
 assert resume['epoch']==epochs-1 and resume['step']==steps and resume['wandb_id']==run.id
 assert all(torch.equal(v,resume['model'][k]) for k,v in saved.items())
 assert all(torch.isfinite(v).all() for st in resume['optimizer']['state'].values() for v in st.values() if torch.is_tensor(v))
 # Tests use the persisted FP16 checkpoint reloaded into FP32 parameters.
 model.load_state_dict(saved,strict=True)
 del saved,fresh,resume;gc.collect();torch.cuda.empty_cache()
 after=testing('after')
 report=dict(complete=True,model=kind,epochs=epochs,total_microsteps=steps*epochs,optimizer_updates=updates,training_seconds=time.time()-started,weights=str(weights),weights_sha256=sha(weights),resume_sha256=sha(resume_path),validation_before=before['validation'],validation_after=after['validation'],facts_before=before['factual_probes']['correct'],facts_after=after['factual_probes']['correct'],wandb_id=run.id,tests_complete=True,finite_weights=True,strict_reload=True,optimizer_finite=True,completed_at=time.time())
 run.summary.update(report);run.finish();write(out/'completion.json',report)
 phase('complete_awaiting_local_export')
 subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(out/'requirements-resolved.txt').open('w'),check=True)
 include=[p for p in out.iterdir() if p.is_file() and p.suffix in ['.json','.jsonl','.log','.txt']]+[out/'wandb']
 hashes={str(p.relative_to(out)):sha(p) for p in include if p.is_file()}
 write(out/'artifact_manifest.json',hashes);include.append(out/'artifact_manifest.json')
 with tarfile.open('/tmp/minimind-sft-'+kind+'-results.tar.gz','w:gz') as tar:
  for p in include:tar.add(p,arcname=str(p.relative_to(out)))
 write(out/'export_ready.json',dict(archive='/tmp/minimind-sft-'+kind+'-results.tar.gz',sha256=sha('/tmp/minimind-sft-'+kind+'-results.tar.gz'),complete=True))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['dense','moe'],required=True);a=p.parse_args()
 try:main(a.kind)
 except BaseException as e:
  write(ROOT/a.kind/'status.json',dict(phase='failed',error=type(e).__name__,detail=str(e)));raise
