"""Independent Dense/MoE pretraining from scratch; immutable durable checkpoints."""
import argparse,gc,json,math,os,shutil,subprocess,sys,tarfile,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,Subset
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from trainer.trainer_utils import setup_seed,SkipBatchSampler
from common import *
class Chunks(Dataset):
 def __init__(self,path,count):self.a=np.memmap(path,dtype=np.uint16,mode='r',shape=(count,LENGTH))
 def __len__(self):return len(self.a)
 def __getitem__(self,i):
  x=torch.from_numpy(self.a[i].astype(np.int64));return x,x.clone()
def main(kind):
 out=ROOT/kind;out.mkdir(parents=True,exist_ok=True)
 for folder in ['out','checkpoints','wandb']:(out/folder).mkdir(exist_ok=True)
 assert not (out/'completion.json').exists()
 RUNTIME.mkdir(exist_ok=True);local_out=RUNTIME/kind;local_out.mkdir(exist_ok=True)
 os.environ.update(WANDB_MODE='offline',WANDB_DIR=str(out/'wandb'),WANDB_CACHE_DIR=str(RUNTIME/'wandb-cache'),HF_HOME=str(RUNTIME/'hf'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
 import wandb
 def status(phase,**kw):
  write(out/'status.json',dict(phase=phase,timestamp=time.time(),**kw));print(phase,json.dumps(kw),flush=True)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
 gpu=torch.cuda.get_device_properties(0);assert '5090' in gpu.name and torch.cuda.is_bf16_supported()
 status('waiting_for_data')
 while not (ROOT/'prepared.json').exists():
  p=ROOT/'prepare_status.json'
  if p.exists() and json.loads(p.read_text()).get('phase')=='failed':raise RuntimeError('Data preparation failed')
  time.sleep(10)
 manifest=json.loads((ROOT/'prepared.json').read_text());assert manifest['max_seq_len']==340 and manifest['stride']==339
 status('verify_and_cache_data')
 for split,record in manifest['files'].items():
  local=RUNTIME/f'{split}.bin'
  if not local.exists() or local.stat().st_size!=record['bytes']:shutil.copyfile(record['path'],local)
  assert sha(local)==record['sha256']
 train=Chunks(RUNTIME/'train.bin',manifest['splits']['train']['chunks']);valid=Chunks(RUNTIME/'validation.bin',manifest['splits']['validation']['chunks'])
 indices=np.random.default_rng(42).choice(len(valid),size=min(1024,len(valid)),replace=False).tolist();validation=Subset(valid,indices)
 tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True);assert len(tok)==6400
 config=MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=kind=='moe');config.to_json_file(str(out/'config.json'))
 setup_seed(42);model=MiniMindForCausalLM(config).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
 batch=64;accum=4;epochs=2;steps=math.ceil(len(train)/batch);total_updates=epochs*math.ceil(steps/accum);warm=min(250,max(1,int(total_updates*.02)))
 settings=dict(model=kind,epochs=epochs,batch_size=batch,accumulation_steps=accum,effective_batch_size=256,max_seq_len=340,learning_rate=3e-4,min_learning_rate=3e-5,warmup_updates=warm,dtype='bfloat16',grad_clip=1.,weight_decay=.01,seed=42,from_weight='none',steps_per_epoch=steps,optimizer_updates=total_updates,training_chunks=len(train),training_targets_per_epoch=len(train)*339,validation_chunks=len(validation),validation_indices=indices,model_parameters=sum(p.numel() for p in model.parameters()),gpu=gpu.name,torch=torch.__version__,cuda=torch.version.cuda,source_sha256=manifest['source_sha256'],data_sha256=manifest['files']['train']['sha256'],policy=manifest['policy'],checkpoint_interval_updates=2000,checkpoint_retention=2)
 write(out/'training_settings.json',settings)
 run=wandb.init(entity='mind4u',project=PROJECT,name=f'RTX5090-{kind}-full-doc-drop-tail-2epoch-20261024',dir=str(out/'wandb'),config=settings,mode='offline')
 write(out/'wandb_run.json',dict(id=run.id,url=f'https://wandb.ai/mind4u/{PROJECT}/runs/{run.id}',project=PROJECT,mode='offline_backed_up_with_local_live_bridge'))
 def log(row):
  row=dict(row,timestamp=time.time());run.log(row)
  with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 def evaluate():
  model.eval();nll=targets=0
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for x,y in DataLoader(validation,batch_size=32,num_workers=2):
    x=x.cuda();y=y.cuda();n=y.shape[0]*(LENGTH-1);loss=float(model(x,labels=y).loss);assert math.isfinite(loss);nll+=loss*n;targets+=n
  model.train();return dict(loss=nll/targets,perplexity=math.exp(nll/targets),tokens=targets,chunks=len(validation))
 status('validation_before');before=evaluate();write(out/'evaluation_before.json',before);log(dict(optimizer_updates=0,validation_before_loss=before['loss']))
 updates=0;started=time.time();suffix='_moe' if kind=='moe' else '';published=[];last=None
 def checkpoint(epoch,step):
  nonlocal last
  status('checkpoint',epoch=epoch+1,step=step,optimizer_updates=updates)
  name=f'update-{updates:08d}';weight=local_out/(name+'-weights.pth');resume=local_out/(name+'-resume.pth')
  cpu={k:v.detach().cpu() for k,v in model.state_dict().items()}
  torch.save({k:v.half() for k,v in cpu.items()},weight)
  torch.save(dict(model=cpu,optimizer=optimizer.state_dict(),epoch=epoch,step=step,updates=updates,wandb_id=run.id,settings=settings),resume)
  record=dict(epoch=epoch+1,step=step,optimizer_updates=updates,wandb_id=run.id,files={})
  for source in [weight,resume]:
   dest=out/'checkpoints'/source.name;shutil.copyfile(source,dest);digest=sha(source);assert sha(dest)==digest
   record['files'][source.name]=dict(path=str(dest),sha256=digest,bytes=dest.stat().st_size)
  write(out/'checkpoint.json',record);published.append((weight,resume));last=(weight,resume,record)
  while len(published)>2:
   older=published.pop(0)
   for p in older:
    (out/'checkpoints'/p.name).unlink();p.unlink()
  del cpu;gc.collect()
 status('training',settings=settings);optimizer.zero_grad(set_to_none=True)
 for epoch in range(epochs):
  setup_seed(42+epoch);order=torch.randperm(len(train),generator=torch.Generator().manual_seed(42+epoch)).tolist()
  loader=DataLoader(train,batch_sampler=SkipBatchSampler(order,batch,0),num_workers=4,pin_memory=True)
  for step,(x,y) in enumerate(loader,1):
   x=x.cuda();y=y.cuda();group_size=min(accum,steps-((step-1)//accum)*accum);next_update=updates+1
   lr=3e-4*next_update/warm if next_update<=warm else 3e-4*(.1+.45*(1+math.cos(math.pi*(next_update-warm)/max(1,total_updates-warm))))
   for group in optimizer.param_groups:group['lr']=lr
   with torch.autocast('cuda',dtype=torch.bfloat16):res=model(x,labels=y);loss=res.loss+res.aux_loss
   assert torch.isfinite(loss);(loss/group_size).backward()
   if step%accum==0 or step==steps:
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    if kind=='moe' and updates==0:
     routers=[p.grad for n,p in model.named_parameters() if n.endswith('mlp.gate.weight')];assert len(routers)==8 and all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in routers)
    optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
    if updates==1 or updates%10==0 or step==steps:
     elapsed=time.time()-started;row=dict(epoch=epoch+1,step=step,steps_per_epoch=steps,optimizer_updates=updates,total_optimizer_updates=total_updates,loss=float(loss.detach()),logits_loss=float(res.loss.detach()),aux_loss=float(res.aux_loss.detach()),learning_rate=lr,grad_norm=float(norm),elapsed_training_seconds=elapsed,eta_hours=elapsed/updates*(total_updates-updates)/3600)
     log(row);status('training',**row);write(out/'progress.json',row)
    if updates==1 or updates%2000==0 or step==steps:checkpoint(epoch,step)
   del x,y,res,loss
  weight=last[0];shutil.copyfile(weight,out/f'out/pretrain_768{suffix}_epoch{epoch+1}.pth')
  v=evaluate();write(out/f'epoch{epoch+1}.json',dict(epoch=epoch+1,optimizer_updates=updates,validation=v));log(dict(optimizer_updates=updates,**{f'validation_epoch{epoch+1}_loss':v['loss']}))
 assert updates==total_updates;status('verify_final')
 weight,resume,receipt=last;saved=torch.load(weight,map_location='cpu',weights_only=True);state=torch.load(resume,map_location='cpu',weights_only=False)
 assert state['epoch']==1 and state['step']==steps and state['updates']==total_updates
 assert all(torch.isfinite(v).all() for v in saved.values())
 assert all(torch.equal(v,state['model'][k].half()) for k,v in saved.items())
 assert all(torch.isfinite(v).all() for item in state['optimizer']['state'].values() for v in item.values() if torch.is_tensor(v))
 fresh=MiniMindForCausalLM(config);fresh.load_state_dict(saved,strict=True);model.load_state_dict(saved,strict=True)
 del fresh,state,saved;gc.collect();torch.cuda.empty_cache()
 target=out/f'out/pretrain_768{suffix}.pth';shutil.copyfile(weight,target);assert sha(target)==sha(weight)
 status('validation_after_reload');after=evaluate();write(out/'evaluation_after.json',after)
 continuations=[];model.eval()
 for prompt in ['The purpose of education is','Water freezes when','A triangle has','There are 3 apples and 4 more apples. The total number of apples is']:
  x=tok(prompt,return_tensors='pt',add_special_tokens=False).input_ids.cuda()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):a=model.generate(inputs=x,max_new_tokens=96,do_sample=False,temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,eos_token_id=2,use_cache=True)
  continuations.append(dict(prompt=prompt,response=tok.decode(a[0,x.shape[1]:],skip_special_tokens=True)))
 write(out/'generations_after.json',continuations)
 report=dict(complete=True,model=kind,epochs=epochs,steps=steps*epochs,optimizer_updates=updates,training_seconds=time.time()-started,weights=str(target),weights_sha256=sha(target),resume=receipt['files'][resume.name]['path'],resume_sha256=sha(resume),validation_before=before,validation_after=after,wandb_id=run.id,wandb_url=f'https://wandb.ai/mind4u/{PROJECT}/runs/{run.id}',finite_weights=True,strict_reload=True,optimizer_finite=True,checkpoint_verified=True,tests_complete=True,completed_at=time.time())
 log(dict(optimizer_updates=updates,validation_after_loss=after['loss']));run.summary.update(report);run.finish();write(out/'completion.json',report);status('complete_awaiting_export')
 subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(out/'requirements-resolved.txt').open('w'),check=True)
 include=[p for p in out.iterdir() if p.is_file() and p.suffix in ['.json','.jsonl','.txt','.log']]+[out/'wandb']
 write(out/'artifact_manifest.json',{p.name:sha(p) for p in include if p.is_file()});include.append(out/'artifact_manifest.json')
 archive=RUNTIME/f'{kind}-results.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in include:tar.add(p,arcname=p.name)
 write(out/'export_ready.json',dict(complete=True,archive=str(archive),sha256=sha(archive)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['dense','moe'],required=True);a=p.parse_args()
 try:main(a.kind)
 except BaseException as e:
  write(ROOT/a.kind/'status.json',dict(phase='failed',type=type(e).__name__,detail=str(e),timestamp=time.time()));raise
