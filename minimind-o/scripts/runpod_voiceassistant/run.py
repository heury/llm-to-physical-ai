"""Pinned, shard-wise native MiniMind-O voice fine-tuning."""
import os,sys,json,time,math,random,hashlib,shutil,subprocess,tarfile,io,gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import librosa
from huggingface_hub import snapshot_download,hf_hub_download,HfApi
from transformers import AutoTokenizer,MimiModel
from snac import SNAC
from codec import split_snac
ROOT=Path('/shared-volume/minimind-o-voiceassistant-20261020')
RUNTIME=Path('/tmp/minimind-o-voiceassistant-20261020')
REPO=RUNTIME/'repo'
sys.path.insert(0,str(REPO))
from model.model_omni import MiniMindOmni,OmniConfig
from dataset.omni_dataset import OmniDataset
from trainer.train_sft_omni import omni_collate_fn
from trainer.trainer_utils import setup_seed
import wandb
REV='65dab707fcdd3d43dcfb834398aa9fed4116be3a'
MODELS={
 'base':('jingyaogong/minimind-3o-pytorch','9e22fb0e51852359ff51bd48d91ec3a345dbd75b'),
 'SenseVoiceSmall':('jingyaogong/SenseVoiceSmall','964f07d8f0e48309a5f37770ac9b5ed2cdca6b91'),
 'mimi':('kyutai/mimi','89091b3e466eb6a9d11e537bf26b144f194978f7'),
 'snac':('hubertsiuzdak/snac_24khz','d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec'),
 'vision':('google/siglip2-base-patch32-256','94dffa8cb1179de3e03f091dbc3917e5d5a9ae84')}
SEQ=1024;BATCH=4;ACCUM=4;TOTAL_ROWS=10000;VAL=128
SHARDS=[0,36,72,108,144,180,216,252,288,324]

def write(name,value):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,ensure_ascii=False,indent=2));q.replace(p)
def phase(name,**kw):write('phase.json',dict(phase=name,updated=time.time(),**kw));print(json.dumps(dict(phase=name,**kw)),flush=True)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(path):return torch.load(path,map_location='cpu',weights_only=False)
def finite(x):
 if isinstance(x,torch.Tensor):return not x.is_floating_point() or bool(torch.isfinite(x).all())
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,(tuple,list)):return all(map(finite,x))
 return True

def convert(shard,snac,mimi):
 chunk=0;out=ROOT/'converted'/f'{shard:05d}-{chunk:03d}.parquet'
 phase('download_convert',shard=shard,total_shards=len(SHARDS))
 f=hf_hub_download('gpt-omni/VoiceAssistant-400K',f'data/train-{shard:05d}-of-00325.parquet',repo_type='dataset',revision=REV,cache_dir=RUNTIME/'dataset-cache')
 stats=dict(shard=shard,source=0,retained=0,rejected=[],audio_seconds=0)
 writer=None;rows=[];source_idx=0
 pf=pq.ParquetFile(f)
 assert pf.metadata.num_rows>=1000
 selected=set(random.Random(42+shard).sample(range(pf.metadata.num_rows),1000))
 write(f'selection/{shard:05d}.json',dict(shard=shard,seed=42+shard,source_file_rows=pf.metadata.num_rows,selected_rows=sorted(selected)))
 def flush():
  nonlocal writer,rows
  if not rows:return
  table=pa.Table.from_pylist(rows)
  if writer is None:writer=pq.ParquetWriter(str(out)+'.tmp',table.schema,compression='zstd')
  writer.write_table(table);rows=[]
 for b in pf.iter_batches(batch_size=8):
  for row in b.to_pylist():
   idx=source_idx;source_idx+=1
   if idx not in selected:continue
   stats['source']+=1
   snac.cuda();mimi.cuda()
   try:
    if not row['question'].strip() or not row['answer'].strip():raise ValueError('empty_text')
    levels=split_snac(row['answer_snac'])
    # SNAC 24k: each coarse frame spans 2048 samples. Decode at most 80s;
    # complete text is retained up to model sequence length. Report this cap.
    capped=len(levels[0])>938
    levels=[x[:938*(2**i)] for i,x in enumerate(levels)]
    with torch.no_grad():
     wav=snac.decode([torch.tensor(x,device='cuda')[None] for x in levels])
     assert wav.ndim==3 and torch.isfinite(wav).all()
     codes=mimi.encode(wav,num_quantizers=8).audio_codes[0]
     assert codes.shape[0]==8 and codes.min()>=0 and codes.max()<2048
    q=row['question_audio'];audio=q.get('bytes')
    if not audio:raise ValueError('missing_question_audio')
    w,sr=sf.read(io.BytesIO(audio));assert len(w)>0 and np.isfinite(w).all()
    # Validate input audio, retain original bytes/sample rate for native loader.
    rows.append(dict(conversations=json.dumps([{'role':'system','content':'You are a helpful voice assistant.'},{'role':'user','content':row['question']},{'role':'assistant','content':row['answer']}],ensure_ascii=False),question_audios=[audio],answer_audios=[codes.T.cpu().flatten().tolist()],source_shard=shard,source_row=idx,source_id=str(row['index']),source_split=str(row['split_name']),answer_audio_capped=capped))
    stats['retained']+=1;stats['audio_seconds']+=wav.shape[-1]/24000
    if shard==0 and chunk==0 and stats['retained']==1:
     sf.write(ROOT/'codec_snac_sample.wav',wav.squeeze().cpu().numpy(),24000)
     with torch.no_grad():rec=mimi.decode(codes[None]).audio_values
     assert torch.isfinite(rec).all()
     sf.write(ROOT/'codec_mimi_sample.wav',rec.squeeze().cpu().numpy(),24000)
     write('codec_smoke.json',dict(snac_lengths=list(map(len,levels)),mimi_shape=list(codes.shape),mimi_min=int(codes.min()),mimi_max=int(codes.max()),seconds=wav.shape[-1]/24000))
   except (ValueError,RuntimeError,AssertionError) as e:
    # Invalid source records are enumerated, never silently skipped.
    if isinstance(e,torch.OutOfMemoryError):raise
    stats['rejected'].append(dict(row=idx,error=str(e)))
   if len(rows)>=16:flush()
   if stats['source']%25==0:write('conversion_progress.json',dict(**stats,updated=time.time()))
   if stats['source']==256:
    flush();assert writer is not None;writer.close();Path(str(out)+'.tmp').replace(out)
    out.with_suffix('.json').write_text(json.dumps(stats,indent=2))
    yield out,stats
    chunk+=1;out=ROOT/'converted'/f'{shard:05d}-{chunk:03d}.parquet';writer=None
    stats=dict(shard=shard,chunk=chunk,source=0,retained=0,rejected=[],audio_seconds=0)
 flush()
 if stats['source']:
  assert writer is not None and stats['retained']>0
  writer.close();Path(str(out)+'.tmp').replace(out)
  out.with_suffix('.json').write_text(json.dumps(stats,indent=2))
  yield out,stats
 shutil.rmtree(RUNTIME/'dataset-cache',ignore_errors=True)

class FixedDataset(OmniDataset):
 def augment_wav(self,wav,sr=16000):return wav
 def augment_mel(self,fbank):return fbank
 def create_chat_prompt(self,conversations,audio_features_length=0):
  messages=[dict(t) for t in conversations]
  if audio_features_length:messages[-2]['content']=self.audio_token*audio_features_length
  return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)

def dataset(path,model,tokenizer):
 return FixedDataset(str(path),tokenizer,audio_processor=model.audio_processor,vision_processor=None,max_length=SEQ,scheduled_sampling=0)

def batch_to_gpu(batch):
 ids,labels,al,ain,lens,pv,spk=batch
 return ids.cuda(),labels.cuda(),al.cuda(),ain.cuda() if ain is not None else None,lens.cuda(),None,spk.cuda()
def loss_on(model,batch):
 ids,labels,al,ain,lens,pv,spk=batch_to_gpu(batch)
 assert (labels!=-100).sum()>0 and (al!=-100).sum()>0
 with torch.autocast('cuda',dtype=torch.bfloat16):
  res=model(ids,audio_inputs=ain,audio_lens=lens,spk_emb=spk)
  tl=F.cross_entropy(res.logits.flatten(0,1),labels.flatten(),ignore_index=-100)
  au=tl.new_zeros(())
  counts=[]
  for i,logits in enumerate(res.audio_logits):
   target=al[:,i,:].reshape(-1);valid=target!=-100
   raw=F.cross_entropy(logits.flatten(0,1),target,ignore_index=-100,reduction='none')
   au=au+(raw*(1+9*(target==2050))).sum()/valid.sum().clamp_min(1)
   counts.append(int(valid.sum()))
  au=au/8;loss=tl+au+res.aux_loss
 assert torch.isfinite(loss),loss
 return loss,tl,au,int((labels!=-100).sum()),counts

@torch.no_grad()
def evaluate(model,ds,name):
 model.eval();sums=[0.,0.];n=[0,0];rows=0
 for batch in torch.utils.data.DataLoader(ds,batch_size=2,collate_fn=omni_collate_fn):
  loss,t,a,tc,ac=loss_on(model,batch);sums[0]+=t.item()*tc;sums[1]+=a.item()*sum(ac);n[0]+=tc;n[1]+=sum(ac);rows+=len(batch[0])
 d=dict(text_loss=sums[0]/n[0],audio_loss=sums[1]/n[1],text_tokens=n[0],audio_tokens=n[1],rows=rows)
 write(name,d);model.train();return d

def checkpoint(model,opt,step,shard,seen,name='sft_omni_768.pth'):
 model.eval();state={k:v.detach().cpu().half() for k,v in model.state_dict().items()}
 assert finite(state) and finite(opt.state_dict())
 p=ROOT/'out'/name;tmp=p.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(p)
 r=ROOT/'out'/'resume.pth';torch.save(dict(model=state,optimizer=opt.state_dict(),step=step,shard=shard,seen=seen,wandb_id=wandb.run.id),r.with_suffix('.tmp'));r.with_suffix('.tmp').replace(r)
 write('checkpoint.json',dict(step=step,shard=shard,seen=seen,sha256=sha(p),wandb_id=wandb.run.id))
 model.train()

def samples(model,tokenizer,mimi):
 model.eval();results=[]
 for i,prompt in enumerate(['Please introduce yourself.','How do I make a cup of tea?']):
  s=tokenizer.apply_chat_template([{'role':'system','content':'You are a helpful voice assistant.'},{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,open_thinking=False)
  ids=torch.tensor([tokenizer.encode(s,add_special_tokens=False)],device='cuda');frames=[];answer=''
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   for y,fr in model.generate(ids,eos_token_id=tokenizer.eos_token_id,max_new_tokens=256,temperature=.3,top_p=.9,stream=True,return_audio_codes=True,open_thinking=False):
    if y is not None:answer=tokenizer.decode(y[0],skip_special_tokens=True)
    if fr is not None:frames.append(fr)
  result=dict(prompt=prompt,answer=answer,frames=len(frames))
  if frames:
   codes=torch.tensor(frames,device='cuda').T[None];assert codes.min()>=0 and codes.max()<2048
   with torch.no_grad():wav=mimi.decode(codes).audio_values
   assert torch.isfinite(wav).all();sf.write(ROOT/f'generated_{i}.wav',wav.squeeze().cpu().numpy(),24000)
  results.append(result)
 write('generation_samples.json',results)

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 for d in ['models','converted','out','tokenizer']:(ROOT/d).mkdir(exist_ok=True)
 assert not (ROOT/'out/resume.pth').exists(),'Investigate partial run before restart; no silent overwrite'
 setup_seed(42);torch.set_num_threads(4)
 write('training_settings.json',dict(dataset='gpt-omni/VoiceAssistant-400K',revision=REV,models=MODELS,epochs=1,batch=BATCH,accumulation=ACCUM,sequence=SEQ,lr=2e-5,seed=42,validation=VAL,selected_source_rows=TOTAL_ROWS,selected_shards=SHARDS,selection='1000 seeded random records per evenly spaced shard',input_mode='audio_only; question text retained in provenance',answer_audio_cap_seconds=80))
 phase('download_models')
 for name,(repo,rev) in MODELS.items():
  snapshot_download(repo,revision=rev,local_dir=ROOT/'models'/name,allow_patterns=['sft_omni_768.pth'] if name=='base' else None,max_workers=4)
 tokenizer=AutoTokenizer.from_pretrained(REPO/'model');tokenizer.save_pretrained(ROOT/'tokenizer')
 snac=SNAC.from_pretrained(str(ROOT/'models/snac')).eval().cuda()
 mimi=MimiModel.from_pretrained(ROOT/'models/mimi').eval().cuda()
 model=MiniMindOmni(OmniConfig(hidden_size=768,num_hidden_layers=8),audio_encoder_path=str(ROOT/'models/SenseVoiceSmall'),vision_model_path=str(ROOT/'models/vision'))
 base=load(ROOT/'models/base/sft_omni_768.pth');model.load_state_dict(base,strict=True);del base
 assert model.audio_encoder is not None and model.audio_processor is not None
 model=model.cuda();model.audio_encoder.cuda();model.audio_encoder.eval()
 # No image examples: preserve complete vision path while freezing unused projector.
 for p in model.vision_proj.parameters():p.requires_grad=False
 if model.vision_encoder is not None:model.vision_encoder.cpu()
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5)
 run=wandb.init(project='MiniMind-O-VoiceAssistant-400K',name='A40-VoiceAssistant-10k-1epoch',config=json.loads((ROOT/'training_settings.json').read_text()),dir=str(ROOT))
 write('wandb_run.json',dict(id=run.id,url=run.url))
 step=seen=source_total=retained=0;manifest=[];valds=None;start=time.time()
 for shard in SHARDS:
  for path,stats in convert(shard,snac,mimi):
   snac.cpu();mimi.cpu();torch.cuda.empty_cache()
   source_total+=stats['source'];retained+=stats['retained'];manifest.append(stats)
   ds=dataset(path,model,tokenizer)
   # Filter examples with no answer labels after audio prefix truncation.
   usable=[];excluded=[]
   for i in range(len(ds)):
    item=ds[i]
    if int((item[1]!=-100).sum())>=8 and int((item[2]!=-100).sum())>=8:usable.append(i)
    else:excluded.append(i)
   stats['sequence_excluded']=excluded
   if shard==0 and valds is None:
    assert len(usable)>VAL+BATCH
    valds=torch.utils.data.Subset(ds,usable[:VAL]);usable=usable[VAL:]
    write('validation_indices.json',dict(shard=0,rows=valds.indices))
    phase('smoke')
    batch=omni_collate_fn([ds[i] for i in usable[:BATCH]])
    model.train();opt.zero_grad(set_to_none=True);loss,tl,au,tc,ac=loss_on(model,batch);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    assert torch.isfinite(norm) and norm>0
    assert model.audio_proj.mlp[1].weight.grad is not None and model.audio_proj.mlp[1].weight.grad.abs().sum()>0
    write('smoke.json',dict(loss=loss.item(),text_loss=tl.item(),audio_loss=au.item(),grad_norm=norm.item(),text_tokens=tc,audio_tokens=ac,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    opt.zero_grad(set_to_none=True)
    evaluate(model,valds,'evaluation_before.json')
   stats['train_rows']=len(usable);write('dataset_manifest.json',dict(revision=REV,source_rows=source_total,retained_rows=retained,shards=manifest))
   loader=torch.utils.data.DataLoader(torch.utils.data.Subset(ds,usable),batch_size=BATCH,shuffle=True,num_workers=0,collate_fn=omni_collate_fn)
   phase('training',shard=shard,total_shards=len(SHARDS))
   model.train();opt.zero_grad(set_to_none=True)
   for micro,batch in enumerate(loader):
    # Normalize final accumulation group exactly, including shard boundary.
    group_size=min(ACCUM,len(loader)-(micro//ACCUM)*ACCUM)
    loss,tl,au,tc,ac=loss_on(model,batch);(loss/group_size).backward();seen+=len(batch[0])
    if (micro+1)%ACCUM==0 or micro+1==len(loader):
     norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
     assert torch.isfinite(norm)
     lr=2e-5*(.1+.45*(1+math.cos(math.pi*min(seen/(TOTAL_ROWS-VAL),1))));step+=1
     for g in opt.param_groups:g['lr']=lr
     opt.step();opt.zero_grad(set_to_none=True)
     metrics=dict(step=step,shard=shard,seen=seen,loss=loss.item(),text_loss=tl.item(),audio_loss=au.item(),grad_norm=norm.item(),lr=lr,elapsed_seconds=time.time()-start)
     assert all(math.isfinite(v) for v in metrics.values())
     run.log(metrics,step=step);write('progress.json',metrics)
     if step%10==0 or step==1:print(json.dumps(metrics),flush=True)
     if step==1 or step%250==0:checkpoint(model,opt,step,shard,seen)
     if step==1:
      saved=load(ROOT/'out/sft_omni_768.pth');original=load(ROOT/'models/base/sft_omni_768.pth')
      changed=[k for k in saved if not torch.equal(saved[k],original[k].half())]
      assert any(k.startswith('audio_proj.') for k in changed) and any(k.startswith('talker.') for k in changed)
      write('first_checkpoint_validation.json',dict(step=step,changed_tensors=len(changed),finite=finite(saved),wandb_id=run.id,sha256=sha(ROOT/'out/sft_omni_768.pth')));del saved,original
   checkpoint(model,opt,step,shard,seen)
   del loader,ds;gc.collect()
 assert source_total==TOTAL_ROWS,(source_total,TOTAL_ROWS)
 phase('final_validation');evaluation=evaluate(model,valds,'evaluation_after.json')
 state=load(ROOT/'out/sft_omni_768.pth');model.load_state_dict(state,strict=True);assert finite(state)
 evaluation.update(step=step,seen=seen,source_rows=source_total,checkpoint_sha256=sha(ROOT/'out/sft_omni_768.pth'));write('evaluation_after.json',evaluation)
 mimi.cuda();samples(model,tokenizer,mimi)
 run.summary.update(evaluation);run.finish()
 phase('packaging')
 subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(ROOT/'requirements-resolved.txt').open('w'),check=True)
 from package import finalize
 finalize(ROOT,REPO)
if __name__=='__main__':main()
