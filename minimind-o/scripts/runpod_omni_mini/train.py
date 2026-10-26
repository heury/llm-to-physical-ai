"""Full-epoch staged fine-tuning with heldouts and native Omni checkpoints."""
import os, sys, json, time, math, random, io, zipfile, hashlib, shutil, traceback, gc
from pathlib import Path
ROOT=Path(os.environ.get('OMNI_RUN_ROOT','/workspace/omni-mini-20261025'))
REPO=ROOT/'repo'
sys.path.insert(0,str(REPO))
import numpy as np
import torch
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image,ImageOps
from transformers import AutoTokenizer
from torch.utils.data import DataLoader,Subset
from model.model_omni import MiniMindOmni,OmniConfig
from dataset.omni_dataset import OmniDataset
from trainer.train_sft_omni import omni_collate_fn
from trainer.trainer_utils import setup_seed
from prepare import REVISIONS
ART=ROOT/'artifacts';ART.mkdir(parents=True,exist_ok=True)

def write(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(p)
def phase(name,**kw):
 obj=dict(phase=name,time=time.time(),**kw);write(ROOT/'status.json',obj);print(json.dumps(obj),flush=True)
def sha(path):
 with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class StableDataset(OmniDataset):
 """Keep native token/label alignment, but disable audio corruption and random prompts."""
 def augment_wav(self,wav,sr=16000):return wav
 def augment_mel(self,fbank):return fbank
 def create_chat_prompt(self,conversations,audio_features_length=0):
  messages=[dict(t) for t in conversations]
  if audio_features_length:
   last=max(i for i,t in enumerate(messages) if t['role']=='user')
   messages[last]['content']=self.audio_token*audio_features_length
  return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
 def load_image_inputs(self,image_bytes):
  if isinstance(image_bytes,str):
   archive,member=image_bytes.split('::',1)
   if not hasattr(self,'archives'):self.archives={}
   if archive not in self.archives:self.archives[archive]=zipfile.ZipFile(ROOT/'data/vision'/archive)
   image=ImageOps.exif_transpose(Image.open(io.BytesIO(self.archives[archive].read(member)))).convert('RGB')
   return dict(self.vision_processor(images=image,return_tensors='pt'))
  return super().load_image_inputs(image_bytes)
 def __getitem__(self,i):
  # Repeatable per-row multi-turn selection and reference drop across evaluation passes.
  state=random.getstate();random.seed(42+int(i))
  try:return super().__getitem__(i)
  finally:random.setstate(state)

def vision_table(split):
 rows=[]
 for p in sorted((ROOT/'data/vision/conversations'/split).glob('*.parquet')):
  for row in pq.read_table(p).to_pylist():
   rows.append(dict(conversations=json.dumps([{'role':{'human':'user','gpt':'assistant'}[t['from']],'content':t['value']} for t in row['conversations']],ensure_ascii=False),image_bytes=row['image_archive']+'::'+row['image'],source_id=row['id']))
 assert rows,split
 path=ROOT/'converted'/f'{split}.parquet';path.parent.mkdir(exist_ok=True)
 pq.write_table(pa.Table.from_pylist(rows),path)
 return path

def make_dataset(path,model,tok,kind):
 return StableDataset(str(path),tok,audio_processor=model.audio_processor if kind=='a2a' else None,vision_processor=model.vision_processor if kind=='vision' else None,max_length=1024 if kind!='vision' else 384,scheduled_sampling=0)

def collate(batch):
 b=list(omni_collate_fn(batch))
 # Remove only trailing columns with no supervised target in the entire batch.
 valid=(b[1]!=-100)|(b[2]!=-100).any(1)
 indices=valid.nonzero()
 end=max(2,int(indices[:,1].max())+1) if len(indices) else b[0].shape[-1]
 b[0]=b[0][...,:end];b[1]=b[1][...,:end];b[2]=b[2][...,:end]
 return b

def forward_loss(model,batch):
 ids,labels,al,ain,lens,pv,spk=batch
 ids=ids.cuda();labels=labels.cuda();al=al.cuda()
 kwargs=dict(audio_inputs=ain.cuda() if ain is not None else None,audio_lens=lens.cuda(),pixel_values={k:v.cuda() for k,v in pv.items()} if pv is not None else None,spk_emb=spk.cuda())
 nt=(labels!=-100).sum();na=(al!=-100).sum()
 if not nt:return None
 with torch.autocast('cuda',dtype=torch.bfloat16):
  res=model(ids,**kwargs)
  ts=F.cross_entropy(res.logits.flatten(0,1),labels.flatten(),ignore_index=-100,reduction='sum')
  tl=ts/nt
  audio_sum=tl.new_zeros(());audio_loss=tl.new_zeros(())
  for i,logits in enumerate(res.audio_logits):
   target=al[:,i,:].reshape(-1);n=(target!=-100).sum()
   if n:
    raw=F.cross_entropy(logits.flatten(0,1),target,ignore_index=-100,reduction='none')
    weighted=(raw*(1+9*(target==2050))).sum()
    audio_sum=audio_sum+weighted;audio_loss=audio_loss+weighted/n
  loss=tl+audio_loss/8+res.aux_loss
 assert torch.isfinite(loss),'Nonfinite loss'
 return loss,dict(text_sum=float(ts.detach()),text_tokens=int(nt),audio_sum=float(audio_sum.detach()),audio_tokens=int(na),examples=len(ids),zero_text_examples=int(((labels!=-100).sum(1)==0).sum()),audio_stop_targets=int((al==2050).sum()))

def metrics(stats):
 return dict(text_loss=stats['text_sum']/max(stats['text_tokens'],1),audio_weighted_loss=stats['audio_sum']/max(stats['audio_tokens'],1),**stats)

def validate(model,ds,tag,batch=4):
 model.eval();s=dict(text_sum=0.,text_tokens=0,audio_sum=0.,audio_tokens=0,examples=0,zero_text_examples=0,audio_stop_targets=0)
 with torch.no_grad():
  for b in DataLoader(ds,batch_size=batch,shuffle=False,num_workers=2,collate_fn=collate):
   result=forward_loss(model,b)
   if result is None:raise RuntimeError('Validation batch has no targets')
   for k,v in result[1].items():s[k]+=v
 result=metrics(s);result['tag']=tag
 write(ART/'metrics'/f'{tag}.json',result);print('VALIDATION',json.dumps(result),flush=True)
 return result

def save(model,path):
 tmp=Path(str(path)+'.tmp')
 state={k:v.detach().cpu().half() for k,v in model.state_dict().items()}
 assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
 torch.save(state,tmp);tmp.replace(path)

def train_stage(model,ds,name,mode,lr,batch,accum):
 completed=ART/'metrics'/f'{name}_training.json'
 if completed.exists() and (ROOT/f'{name}.pth').exists():
  model.load_state_dict(torch.load(ROOT/f'{name}.pth',map_location='cpu',weights_only=True),strict=True)
  phase('resume_completed_stage',stage=name)
  return json.loads(completed.read_text())
 setup_seed(42)
 for p in model.parameters():p.requires_grad_(mode=='all')
 # Do not drift the unused vision projector during voice stages.
 if mode=='all':
  for p in model.vision_proj.parameters():p.requires_grad_(False)
 else:
  for p in getattr(model,mode).parameters():p.requires_grad_(True)
 params=[p for p in model.parameters() if p.requires_grad]
 optimizer=torch.optim.AdamW(params,lr=lr)
 loader=DataLoader(ds,batch_size=batch,shuffle=True,num_workers=4,collate_fn=collate,pin_memory=True)
 n=len(loader);model.train();start=time.time();updates=0;seen=0;skipped=0
 trainable=sum(p.numel() for p in params)
 phase('training',stage=name,batches=n,rows=len(ds),trainable=trainable)
 optimizer.zero_grad(set_to_none=True)
 for step,b in enumerate(loader,1):
  factor=.1+.45*(1+math.cos(math.pi*(step-1)/max(n-1,1)))
  for group in optimizer.param_groups:group['lr']=lr*factor
  result=forward_loss(model,b)
  if result is None:skipped+=len(b[0]);continue
  loss,stat=result
  group_size=min(accum,n-((step-1)//accum)*accum)
  (loss/group_size).backward();seen+=stat['examples']
  if step%accum==0 or step==n:
   norm=torch.nn.utils.clip_grad_norm_(params,1.)
   assert torch.isfinite(norm),'Nonfinite gradient'
   optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
  if step%25==0 or step==1 or step==n:
   record=dict(stage=name,step=step,total_steps=n,examples=seen,loss=float(loss.detach()),**{k:v for k,v in metrics(stat).items() if k!='examples'},lr=lr*factor,elapsed_seconds=time.time()-start,eta_seconds=(time.time()-start)/step*(n-step),gpu_max_gb=torch.cuda.max_memory_allocated()/2**30)
   with (ART/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   phase('training',**record)
  if step%1000==0:
   save(model,ROOT/'latest.pth')
  del result,loss,b
 save(model,ROOT/f'{name}.pth')
 summary=dict(stage=name,mode=mode,learning_rate=lr,epochs=1,rows=len(ds),processed=seen,skipped_no_labels=skipped,batch_size=batch,accumulation_steps=accum,optimizer_updates=updates,trainable_parameters=trainable,elapsed_seconds=time.time()-start)
 write(ART/'metrics'/f'{name}_training.json',summary)
 del optimizer,loader;gc.collect();torch.cuda.empty_cache()
 return summary

def main():
 torch.set_num_threads(4);setup_seed(42)
 assert (ROOT/'PREPARED').exists()
 tok=AutoTokenizer.from_pretrained(REPO/'model')
 model=MiniMindOmni(OmniConfig(hidden_size=768,num_hidden_layers=8),audio_encoder_path=str(ROOT/'models/audio'),vision_model_path=str(ROOT/'models/vision'))
 assert model.audio_encoder is not None and model.vision_encoder is not None
 model.load_state_dict(torch.load(ROOT/'base/sft_omni_768.pth',map_location='cpu',weights_only=True),strict=True)
 model.cuda();model.audio_encoder.cuda().eval();model.vision_encoder.cuda().eval()
 tok.save_pretrained(ART/'tokenizer');model.config.save_pretrained(ART)
 write(ART/'run_config.json',dict(base='jingyaogong/minimind-3o-pytorch/sft_omni_768.pth',base_sha256=sha(ROOT/'base/sft_omni_768.pth'),revisions=REVISIONS,seed=42,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,precision='bfloat16 autocast, FP32 trainable weights',audio_sequence_length=1024,vision_sequence_length=384,audio_stop_loss_weight=10,scheduled_sampling=0,audio_augmentation=False,a2a_prompt='audio-only',vision_training='projector only',pod='zuq0hsnyvgsmjz'))
 shutil.copyfile(ROOT/'revisions.json',ART/'revisions.json')
 splits={};summaries=[];datasets={}
 for kind in ['t2a','a2a']:
  phase('load_dataset',dataset=kind)
  ds=make_dataset(ROOT/f'data/audio/sft_{kind}_mini.parquet',model,tok,kind)
  order=np.random.default_rng(42).permutation(len(ds));val=order[:128].tolist();train=order[128:].tolist()
  splits[kind]=dict(total=len(ds),train=train,validation=val)
  datasets[kind]=(ds,Subset(ds,train),Subset(ds,val))
  validate(model,datasets[kind][2],f'base_{kind}')
 write(ART/'audio_splits.json',splits)
 # All training rows in each requested mini file, apart from heldout rows.
 for name,kind,mode,lr in [('t2a','t2a','all',2e-5),('a2a_align','a2a','audio_proj',1e-4),('a2a','a2a','all',2e-5)]:
  summaries.append(train_stage(model,datasets[kind][1],name,mode,lr,16,2))
  validate(model,datasets[kind][2],f'after_{name}')
 # The supplied visual release already excludes heldout images across stages.
 for name,split,vs in [('vision_align','pretrain_train','pretrain_validation'),('vision_sft','sft_train','sft_validation')]:
  phase('convert_vision',split=split)
  train=make_dataset(vision_table(split),model,tok,'vision');val=make_dataset(vision_table(vs),model,tok,'vision')
  validate(model,val,f'before_{name}')
  summaries.append(train_stage(model,train,name,'vision_proj',5e-5,16,2))
  validate(model,val,f'after_{name}')
  del train,val;gc.collect()
 for kind in ['t2a','a2a']:validate(model,datasets[kind][2],f'final_{kind}')
 test=make_dataset(vision_table('sft_test'),model,tok,'vision');validate(model,test,'final_vision_test')
 final=ART/'sft_omni_768.pth';save(model,final)
 loaded=torch.load(final,map_location='cpu',weights_only=True);model.load_state_dict(loaded,strict=True);del loaded
 write(ART/'training_summary.json',dict(stages=summaries,final_checkpoint_sha256=sha(final),validation_note='Audio row-heldout; potential overlap with base pretraining is unknown. Visual splits from the provided dataset are preserved. Loss metrics are teacher-forced, not speech-quality scores.'))
 phase('trained',checkpoint_sha256=sha(final))
 (ROOT/'TRAINED').write_text(str(time.time()))

if __name__=='__main__':
 try:main()
 except BaseException:
  write(ROOT/'failure.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
