"""Full GSM8K SFT from verified UltraChat checkpoint, test before/after, export."""
import argparse,gc,json,math,os,shutil,subprocess,sys,tarfile,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from trainer.trainer_utils import setup_seed
from common import *
from evaluate import Arrays,evaluate,validation

def main(kind):
 out=ROOT/kind;out.mkdir(parents=True,exist_ok=True);(out/'out').mkdir(exist_ok=True)
 assert not (out/'completion.json').exists()
 RUNTIME.mkdir(exist_ok=True)
 reuse_baseline=os.environ.get('GSM8K_REUSE_BASELINE')=='1'
 wandb_dir=out/('wandb_verified' if reuse_baseline else 'wandb')
 os.environ.update(WANDB_MODE='offline',WANDB_DIR=str(wandb_dir),WANDB_CACHE_DIR=str(RUNTIME/'wandb-cache'),HF_HOME=str(RUNTIME/'hf'),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
 import wandb
 def status(phase,**kw):
  write(out/'status.json',dict(phase=phase,timestamp=time.time(),**kw));print(phase,json.dumps(kw),flush=True)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
 gpu=torch.cuda.get_device_properties(0);assert '5090' in gpu.name
 status('waiting_for_dataset')
 while not (ROOT/'prepared.json').exists():
  if (ROOT/'prepare.exit').exists() and (ROOT/'prepare.exit').read_text().strip()!='0':raise RuntimeError('Dataset preparation failed')
  time.sleep(5)
 manifest=json.loads((ROOT/'prepared.json').read_text())
 for name,digest in manifest['arrays'].items():
  shutil.copyfile(ROOT/name,RUNTIME/name);assert sha(RUNTIME/name)==digest
 assert sha(ROOT/'test-records.json')==manifest['test_records_sha256']
 train=Arrays('train');tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True);assert len(tok)==6400
 base=json.loads((BASE/kind/'completion.json').read_text());weights=Path(base['weights']);assert sha(weights)==base['weights_sha256']
 config=MiniMindConfig(**json.loads((BASE/kind/'config.json').read_text()));assert config.use_moe==(kind=='moe')
 config.to_json_file(str(out/'config.json'));setup_seed(42)
 model=MiniMindForCausalLM(config).cuda();model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
 batch=64;accum=2;epochs=2;microsteps=math.ceil(len(train)/batch);total_updates=epochs*math.ceil(microsteps/accum);peak=5e-5;warm=max(1,int(total_updates*.03))
 optimizer=torch.optim.AdamW(model.parameters(),lr=peak)
 settings=dict(model=kind,epochs=epochs,batch_size=batch,accumulation_steps=accum,effective_batch_size=batch*accum,learning_rate=peak,min_learning_rate=peak*.1,warmup_updates=warm,max_seq_len=LENGTH,dtype='bfloat16',grad_clip=1.,seed=42,steps_per_epoch=microsteps,optimizer_updates=total_updates,training_windows=len(train),training_original_rows=7473,test_rows=1319,base_weights=str(weights),base_sha256=base['weights_sha256'],dataset='openai/gsm8k',dataset_revision=REV,model_parameters=sum(p.numel() for p in model.parameters()),gpu=gpu.name,torch=torch.__version__,checkpoint_io='local staging, then verified immutable shared-volume publication',baseline_reused=reuse_baseline)
 write(out/'training_settings.json',settings)
 run=wandb.init(entity='mind4u',project='MiniMind-GSM8K-SFT',name=f'RTX5090-{kind}-2epoch-seq340-20261024',config=settings,dir=str(wandb_dir),mode='offline')
 write(out/'wandb_run.json',dict(id=run.id,url=f'https://wandb.ai/mind4u/MiniMind-GSM8K-SFT/runs/{run.id}',mode='offline_pending_local_sync'))
 status('tests_before')
 before=json.loads((out/'tests_before.json').read_text()) if reuse_baseline else evaluate(model,tok,out,'before',status)
 if reuse_baseline:assert before['revision']==REV and before['test_rows']==1319
 run.log({'validation/before_loss':before['validation']['loss'],'gsm8k/before_strict_accuracy':before['strict_accuracy'],'gsm8k/before_flexible_accuracy':before['flexible_accuracy']})
 suffix='_moe' if kind=='moe' else '';target=RUNTIME/f'gsm8k_sft_768{suffix}.pth';resume=RUNTIME/'resume.pth';updates=0
 def checkpoint(epoch,step):
  state={k:v.detach().half().cpu() for k,v in model.state_dict().items()}
  torch.save(state,target.with_suffix('.tmp'));target.with_suffix('.tmp').replace(target)
  torch.save(dict(model=state,optimizer=optimizer.state_dict(),epoch=epoch,step=step,updates=updates,wandb_id=run.id,seed=42),resume.with_suffix('.tmp'));resume.with_suffix('.tmp').replace(resume)
 status('training',settings=settings);started=time.time();optimizer.zero_grad(set_to_none=True)
 for epoch in range(epochs):
  setup_seed(42+epoch);loader=DataLoader(train,batch_size=batch,shuffle=True,num_workers=2,pin_memory=True)
  model.train();running_nll=running_tokens=0
  for step,(x,y) in enumerate(loader,1):
   x=x.cuda();y=y.cuda();boundary=step%accum==0 or step==microsteps
   # Correct normalization of the final partial accumulation group.
   group_start=((step-1)//accum)*accum;group_size=min(accum,microsteps-group_start)
   next_update=updates+1;lr=peak*next_update/warm if next_update<=warm else peak*(.1+.45*(1+math.cos(math.pi*(next_update-warm)/max(1,total_updates-warm))))
   for group in optimizer.param_groups:group['lr']=lr
   with torch.autocast('cuda',dtype=torch.bfloat16):
    result=model(x,labels=y);loss=result.loss+result.aux_loss
   assert torch.isfinite(loss);(loss/group_size).backward()
   n=int((y[:,1:]!=-100).sum());running_nll+=float(result.loss.detach())*n;running_tokens+=n
   if boundary:
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    if kind=='moe' and updates==0:
     routers=[p.grad for name,p in model.named_parameters() if name.endswith('mlp.gate.weight')]
     assert len(routers)==8 and all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in routers)
    optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
    row=dict(epoch=epoch+1,step=step,steps_per_epoch=microsteps,optimizer_updates=updates,loss=float(loss.detach()),learning_rate=lr,grad_norm=float(norm),elapsed_training_seconds=time.time()-started)
    run.log(row)
    with (out/('metrics_verified.jsonl' if reuse_baseline else 'metrics.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    if updates==1 or updates%10==0:status('training',**row)
    if updates==1 or updates%100==0 or step==microsteps:checkpoint(epoch,step)
   del result,loss,x,y
  shutil.copyfile(target,out/f'out/gsm8k_sft_768{suffix}_epoch{epoch+1}.pth')
  v=validation(model);write(out/f'epoch{epoch+1}.json',dict(validation=v,train_loss=running_nll/running_tokens));run.log({f'validation/epoch{epoch+1}_loss':v['loss']})
 training_seconds=time.time()-started;assert updates==total_updates
 status('verify_checkpoint');saved=torch.load(target,map_location='cpu',weights_only=True);assert all(torch.isfinite(v).all() for v in saved.values())
 fresh=MiniMindForCausalLM(config);fresh.load_state_dict(saved,strict=True)
 state=torch.load(resume,map_location='cpu',weights_only=False)
 assert state['epoch']==epochs-1 and state['step']==microsteps and state['updates']==updates
 assert all(torch.equal(v,state['model'][k]) for k,v in saved.items())
 assert all(torch.isfinite(v).all() for item in state['optimizer']['state'].values() for v in item.values() if torch.is_tensor(v))
 model.load_state_dict(saved,strict=True);del fresh,state,saved;gc.collect();torch.cuda.empty_cache()
 # Publish to fresh filenames after local round-trip verification, avoiding overwrite cache.
 published=out/f'out/gsm8k_sft_768{suffix}_verified.pth';published_resume=out/'resume_verified.pth'
 shutil.copyfile(target,published);shutil.copyfile(resume,published_resume)
 assert sha(target)==sha(published) and sha(resume)==sha(published_resume)
 target=published;resume=published_resume
 status('tests_after');after=evaluate(model,tok,out,'after',status)
 report=dict(complete=True,model=kind,epochs=epochs,total_microsteps=microsteps*epochs,optimizer_updates=updates,training_seconds=training_seconds,weights=str(target),weights_sha256=sha(target),resume_sha256=sha(resume),validation_before=before['validation'],validation_after=after['validation'],strict_correct_before=before['strict_correct'],strict_correct_after=after['strict_correct'],flexible_correct_before=before['flexible_correct'],flexible_correct_after=after['flexible_correct'],test_rows=1319,wandb_id=run.id,tests_complete=True,finite_weights=True,strict_reload=True,optimizer_finite=True,completed_at=time.time())
 run.log({'validation/after_loss':after['validation']['loss'],'gsm8k/after_strict_accuracy':after['strict_accuracy'],'gsm8k/after_flexible_accuracy':after['flexible_accuracy']});run.summary.update(report);run.finish();write(out/'completion.json',report)
 status('complete_awaiting_local_export')
 subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(out/'requirements-resolved.txt').open('w'),check=True)
 include=[p for p in out.iterdir() if p.is_file() and p.suffix in ['.json','.jsonl','.txt','.log']]+[wandb_dir]
 write(out/'artifact_manifest.json',{str(p.relative_to(out)):sha(p) for p in include if p.is_file()});include.append(out/'artifact_manifest.json')
 archive='/tmp/minimind-gsm8k-'+kind+'-results.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in include:tar.add(p,arcname='wandb' if p==wandb_dir else str(p.relative_to(out)))
 write(out/'export_ready.json',dict(archive=archive,sha256=sha(archive),complete=True))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['dense','moe'],required=True);args=parser.parse_args()
 try:main(args.kind)
 except BaseException as error:
  write(ROOT/args.kind/'status.json',dict(phase='failed',type=type(error).__name__,detail=str(error)));raise
