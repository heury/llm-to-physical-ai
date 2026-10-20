"""Pinned ShareGPT4V captions + image mirror; native MiniMind-V, actual W&B."""
import hashlib, io, json, math, os, random, shutil, subprocess, tarfile, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, VLMConfig

ROOT = Path('/shared-volume/minimind-v-sharegpt4v-20261020')
RUNTIME = Path('/tmp/minimind-v-sharegpt4v-20261020')
BASE_ROOT = Path('/shared-volume/minimind-sft-ultrachat-200k-20261020')
REV = '55d02b0bc53a2754095a14110dda6daedd95671d'
IMAGE_REV = '3bf5b9f3027e298f6a02abab692aed9ead8dc8b2'
VISION_REV = '94dffa8cb1179de3e03f091dbc3917e5d5a9ae84'
SETTINGS = dict(dataset='Lin-Chen/ShareGPT4V', revision=REV, config='ShareGPT4V',
 image_mirror='wooj1nBot/sharegpt4v_flat_dataset', image_revision=IMAGE_REV,
 vision_model='google/siglip2-base-patch32-256', vision_revision=VISION_REV,
 epochs_alignment=1, epochs_sft=1, batch_size=16, accumulation_steps=2,
 sequence_length=768, learning_rate_alignment=4e-4, learning_rate_sft=5e-6,
 seed=42, hidden_size=768, num_hidden_layers=8, save_interval=250,
 image_storage='RGB resized to 256x256 BICUBIC, PNG lossless; SigLIP resize geometry',
 validation_rows=256, image_filename_aliases={'Choi_Min_sik':'Choi_Min-sik','Lee_Byung_hun':'Lee_Byung-hun'})

def write(name, value):
 p=ROOT/name; t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2));t.replace(p)
def phase(name, **extra): write('phase.json',dict(phase=name,updated=time.time(),**extra));print(name,extra,flush=True)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(value,path):
 torch.save(value,str(path)+'.tmp');os.replace(str(path)+'.tmp',path)
def prepare():
 from huggingface_hub import hf_hub_download, snapshot_download
 import pyarrow.ipc as ipc
 import requests
 for n in ['data','tokenizer','vision_encoder','out','checkpoints']:(ROOT/n).mkdir(parents=True,exist_ok=True)
 (RUNTIME/'images').mkdir(exist_ok=True)
 write('training_settings.json',SETTINGS);phase('download_caption_and_vision')
 for n in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']:
  p=BASE_ROOT/'tokenizer'/n
  if p.exists():shutil.copyfile(p,ROOT/'tokenizer'/n)
 base=BASE_ROOT/'out/full_sft_768.pth'
 assert sha(base)==json.loads((BASE_ROOT/'evaluation_after.json').read_text())['checkpoint_sha256']
 shutil.copyfile(base,ROOT/'out/llm_768.pth')
 cap=hf_hub_download(SETTINGS['dataset'],'sharegpt4v_instruct_gpt4-vision_cap100k.json',repo_type='dataset',revision=REV,token=False,cache_dir=str(ROOT/'hf-cache'))
 rows=json.loads(Path(cap).read_text());assert len(rows)==102025
 shutil.copyfile(cap,ROOT/'data/original_captions.json')
 snapshot_download(SETTINGS['vision_model'],revision=VISION_REV,allow_patterns=['config.json','preprocessor_config.json','model.safetensors'],local_dir=ROOT/'vision_encoder',token=False)
 tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True)
 assert tok.convert_tokens_to_ids('<|image_pad|>')==12
 # This mirror supplies image bytes only. Original source captions remain untouched.
 import zipfile, concurrent.futures, threading
 byname={}
 for i,row in enumerate(rows):byname.setdefault(Path(row['image']).name,[]).append(i)
 assert len(byname)==len({r['image'] for r in rows}), 'Ambiguous source basenames'
 seen=set()
 sources=[(SETTINGS['image_mirror'],IMAGE_REV,'images.zip',None)]+[
  ('DavidNguyen/ShareGPT4V-Sam','cae31e9453b57f53c04f67cb759ec9703be3c091',n+'.zip',n)
  for n in ['wikiart','share_textvqa','web-celebrity','web-landmark']]
 write('image_sources.json',[dict(repository=r,revision=v,file=f,prefix=p) for r,v,f,p in sources])
 for repo,revision,filename,prefix in sources:
  phase('prepare_images',source=repo,file=filename,images_verified=len(seen))
  done=ROOT/'data'/(filename+'.json')
  if done.exists():
   previous=json.loads(done.read_text())
   if all((RUNTIME/'images'/f'{i:06d}.png').exists() for i in previous):seen.update(previous);continue
  path=hf_hub_download(repo,filename,repo_type='dataset',revision=revision,token=False,local_dir=RUNTIME/'image-shard')
  with zipfile.ZipFile(path) as z:
   candidates=[]
   aliases={'Choi_Min_sik.jpg':'Choi_Min-sik.jpg','Choi_Min_sik2.jpg':'Choi_Min-sik2.jpg','Lee_Byung_hun.jpg':'Lee_Byung-hun.jpg','Lee_Byung_hun2.jpg':'Lee_Byung-hun2.jpg'}
   for name in z.namelist():
    basename=Path(name).name
    matches=list(byname.get(basename,[]))
    if basename in aliases:matches+=byname.get(aliases[basename],[])
    if prefix:matches=[i for i in matches if rows[i]['image'].startswith(prefix+'/')]
    matches=[i for i in matches if i not in seen]
    if matches:candidates.append((name,matches))
   zip_read_lock=threading.Lock()
   def extract(item):
    name,matches=item
    # Serialize complete ZipFile reads; this runtime races concurrent shared handles.
    with zip_read_lock:raw=z.read(name)
    with Image.open(io.BytesIO(raw)) as image:
     image=ImageOps.exif_transpose(image).convert('RGB').resize((256,256),Image.Resampling.BICUBIC)
     for i in matches:image.save(RUNTIME/'images'/f'{i:06d}.png')
    return matches
   indices=[]
   with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for matched in pool.map(extract,candidates):
     seen.update(matched);indices.extend(matched)
     if len(indices)%2000<4:phase('prepare_images',source=repo,file=filename,images_verified=len(seen))
  write('data/'+filename+'.json',indices)
  Path(path).unlink()
 missing=sorted(set(range(len(rows)))-seen)
 write('image_manifest.json',dict(source=len(rows),verified=len(seen),missing=missing,caption_sha256=sha(ROOT/'data/original_captions.json')))
 assert not missing, f'Missing {len(missing)} original images; full training must not silently use a subset'
 phase('tokenize')
 ids=np.lib.format.open_memmap(ROOT/'data/ids.npy',mode='w+',dtype=np.int16,shape=(len(rows),768))
 labels=np.lib.format.open_memmap(ROOT/'data/labels.npy',mode='w+',dtype=np.int16,shape=(len(rows),768))
 retained=[];excluded=[];truncated=0
 for i,row in enumerate(rows):
  turns=row['conversations'];assert [x['from'] for x in turns]==['human','gpt']
  question=turns[0]['value'];assert question.count('<image>')==1
  messages=[dict(role='system',content='You are a helpful assistant.'),dict(role='user',content=question.replace('<image>','<|image_pad|>'*64)),dict(role='assistant',content=turns[1]['value'])]
  prefix=tok.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True,open_thinking=False)
  full=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,open_thinking=False)
  assert full[:len(prefix)]==prefix
  x=full[:768];y=[-100]*min(len(prefix),len(x))+x[len(prefix):]
  if sum(t!=-100 for t in y[1:])<32:excluded.append(i);continue
  assert x.count(12)==64
  truncated+=len(full)>768
  ids[i]=x+[tok.pad_token_id]*(768-len(x));labels[i]=y+[-100]*(768-len(y));retained.append(i)
 ids.flush();labels.flush()
 order=sorted(retained,key=lambda i:hashlib.sha256(rows[i]['image'].encode()).hexdigest())
 def group(i):return rows[i]['image'].replace('Choi_Min_sik','Choi_Min-sik').replace('Lee_Byung_hun','Lee_Byung-hun')
 val_images=set(list(dict.fromkeys(group(i) for i in order))[:256])
 val=[i for i in order if group(i) in val_images];train=[i for i in order if group(i) not in val_images]
 assert not ({group(i) for i in train}&{group(i) for i in val})
 write('data/splits.json',dict(train=train,validation=val,excluded=excluded))
 write('dataset_manifest.json',dict(source_rows=len(rows),images_verified=len(seen),retained=len(retained),train_rows=len(train),validation_rows=len(val),excluded_short_response=len(excluded),truncated=truncated,source_revision=REV,image_revision=IMAGE_REV,caption_sha256=sha(ROOT/'data/original_captions.json')))
 (ROOT/'PREPARED').write_text('ok\n')

class Data(Dataset):
 def __init__(self,split,processor):
  self.indices=json.loads((ROOT/'data/splits.json').read_text())[split];self.processor=processor
  self.ids=np.load(ROOT/'data/ids.npy',mmap_mode='r');self.labels=np.load(ROOT/'data/labels.npy',mmap_mode='r')
 def __len__(self):return len(self.indices)
 def __getitem__(self,k):
  i=self.indices[k]
  with Image.open(RUNTIME/'images'/f'{i:06d}.png') as image:pixels=self.processor(images=image.convert('RGB'),return_tensors='pt')['pixel_values'][0]
  return torch.tensor(self.ids[i].astype('int64')),torch.tensor(self.labels[i].astype('int64')),pixels

def forward(model,batch):
 x,y,p=(t.cuda(non_blocking=True) for t in batch)
 with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x,labels=y,pixel_values={'pixel_values':p})
 return out.loss, int((y[:,1:]!=-100).sum())

@torch.no_grad()
def evaluate(model,loader):
 model.eval();model.vision_encoder.eval();total=count=0
 for batch in loader:
  loss,n=forward(model,batch);assert torch.isfinite(loss);total+=loss.item()*n;count+=n
 return dict(loss=total/count,perplexity=math.exp(total/count),tokens=count,examples=len(loader.dataset))

def checkpoint(model,opt,stage,step,run):
 weights={k:v.detach().cpu().half() for k,v in model.state_dict().items() if not k.startswith('vision_encoder.')}
 assert all(torch.isfinite(t).all() for t in weights.values())
 save(weights,ROOT/'out'/f'{stage}_vlm_768.pth')
 save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('vision_encoder.')},optimizer=opt.state_dict(),stage=stage,step=step,wandb_id=run.id,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),ROOT/'checkpoints/resume.pth')
 write('checkpoint.json',dict(stage=stage,step=step,wandb_id=run.id,path=str(ROOT/'out'/f'{stage}_vlm_768.pth'),sha256=sha(ROOT/'out'/f'{stage}_vlm_768.pth')))

def train():
 import wandb
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
 torch.cuda.set_per_process_memory_fraction(.90)
 config=VLMConfig(hidden_size=768,num_hidden_layers=8,image_ids=[12],image_token_len=64,max_seq_len=768)
 config.save_pretrained(ROOT)
 model=MiniMindVLM(config,vision_model_path=str(ROOT/'vision_encoder'))
 assert model.vision_encoder is not None and model.processor is not None,'Visual encoder must load'
 result=model.load_state_dict(torch.load(ROOT/'out/llm_768.pth',map_location='cpu',weights_only=True),strict=False)
 assert not result.unexpected_keys
 assert all(k.startswith(('vision_encoder.','vision_proj.')) for k in result.missing_keys)
 model.cuda();model.vision_encoder.eval()
 train_ds=Data('train',model.processor);val_ds=Data('validation',model.processor)
 val_loader=DataLoader(val_ds,batch_size=8,num_workers=2,pin_memory=True)
 run=wandb.init(project='MiniMind-V-ShareGPT4V',name='A40-768-alignment1-sft1',config=SETTINGS,dir=str(ROOT/'wandb'))
 write('wandb_run.json',dict(id=run.id,url=run.url,project=run.project))
 baseline=evaluate(model,val_loader);write('evaluation_before.json',baseline);run.log({'validation/before_loss':baseline['loss']})
 global_step=0;started=time.time()
 for stage,lr in [('pretrain',4e-4),('sft',5e-6)]:
  phase(stage)
  for name,param in model.named_parameters():param.requires_grad=name.startswith('vision_proj.') or (stage=='sft' and name.startswith(('model.layers.0.','model.layers.7.')))
  trainable=[p for p in model.parameters() if p.requires_grad]
  opt=torch.optim.AdamW(trainable,lr=lr);opt.zero_grad(set_to_none=True)
  generator=torch.Generator().manual_seed(42 if stage=='pretrain' else 43)
  loader=DataLoader(train_ds,batch_size=16,shuffle=True,generator=generator,num_workers=4,pin_memory=True)
  frozen_before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad and not n.startswith('vision_encoder.')}
  # Smoke on the longest supervised sample uses the production batch size. No optimizer update.
  batch=next(iter(loader));model.train();model.vision_encoder.eval()
  loss,_=forward(model,batch);loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,1.0,error_if_nonfinite=True)
  assert all(p.grad is None for n,p in model.named_parameters() if not p.requires_grad)
  write(f'smoke_{stage}.json',dict(loss=loss.item(),grad_norm=norm.item(),trainable_parameters=sum(p.numel() for p in trainable),peak_cuda_bytes=torch.cuda.max_memory_allocated(),batch_size=16))
  opt.zero_grad(set_to_none=True)
  # Fresh seeded loader after smoke: all retained training rows appear exactly once.
  generator.manual_seed(42 if stage=='pretrain' else 43)
  for step,batch in enumerate(loader,1):
   model.train();model.vision_encoder.eval()
   rate=lr*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/len(loader))))
   for g in opt.param_groups:g['lr']=rate
   loss,_=forward(model,batch);assert torch.isfinite(loss)
   group_size=min(2,len(loader)-((step-1)//2)*2)
   (loss/group_size).backward()
   norm=None
   if step%2==0 or step==len(loader):
    norm=torch.nn.utils.clip_grad_norm_(trainable,1.0,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
   global_step+=1
   if step==1 or step%10==0 or step==len(loader):
    metrics=dict(stage=stage,step=step,steps=len(loader),global_step=global_step,loss=loss.item(),learning_rate=rate,elapsed_seconds=time.time()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated())
    if norm is not None:metrics['grad_norm']=norm.item()
    print(json.dumps(metrics),flush=True);write('progress.json',metrics)
    run.log({k:v for k,v in metrics.items() if k!='stage'}|{'stage_name':stage},step=global_step)
   if step==1 or step%250==0 or step==len(loader):checkpoint(model,opt,stage,step,run)
  assert all(torch.equal(p.detach().cpu(),frozen_before[n]) for n,p in model.named_parameters() if n in frozen_before),'Frozen language layers changed'
  del frozen_before
  after=evaluate(model,val_loader);write(f'evaluation_{stage}.json',after);run.log({f'validation/{stage}_loss':after['loss']},step=global_step)
  checkpoint(model,opt,stage,len(loader),run)
 final=torch.load(ROOT/'out/sft_vlm_768.pth',map_location='cpu',weights_only=True)
 fresh=MiniMindVLM(config,vision_model_path=str(ROOT/'vision_encoder'))
 incompatible=fresh.load_state_dict(final,strict=False)
 assert not incompatible.unexpected_keys and all(k.startswith('vision_encoder.') for k in incompatible.missing_keys)
 assert all(torch.isfinite(t).all() for t in final.values())
 before=torch.load(ROOT/'out/llm_768.pth',map_location='cpu',weights_only=True)
 changed=[k for k in before if not torch.equal(before[k].half(),final[k])]
 assert changed
 samples=[];model.eval();rows=json.loads((ROOT/'data/original_captions.json').read_text());tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer')
 for j in range(4):
  x,y,p=val_ds[j];start=int(torch.nonzero(y!=-100)[0]);prompt=x[:start].unsqueeze(0).cuda()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   out=model.generate(prompt,pixel_values={'pixel_values':p.unsqueeze(0).cuda()},max_new_tokens=128,do_sample=False,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
  samples.append(dict(image=rows[val_ds.indices[j]]['image'],answer=tok.decode(out[0,prompt.shape[1]:],skip_special_tokens=True),reference=rows[val_ds.indices[j]]['conversations'][1]['value']))
 write('samples.json',samples)
 report=dict(**after,checkpoint_sha256=sha(ROOT/'out/sft_vlm_768.pth'),changed_language_tensors=len(changed),finite_weights=True,strict_nonvision_load=True,global_steps=global_step,seconds=time.time()-started,wandb_id=run.id)
 write('evaluation_after.json',report)
 run.summary.update(report);run.finish()
 subprocess.run([os.sys.executable,'-m','pip','freeze'],stdout=open(ROOT/'requirements-resolved.txt','w'),check=True)
 package()
 (ROOT/'SUCCESS').write_text('verified full training completed\n');phase('complete')

def package():
 bundle=RUNTIME/'bundle';bundle.mkdir(exist_ok=True)
 for n in ['tokenizer','vision_encoder']:
  shutil.copytree(ROOT/n,bundle/n,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.cache'))
 for p in ROOT.glob('*.json'):shutil.copyfile(p,bundle/p.name)
 for n in ['sft_vlm_768.pth','pretrain_vlm_768.pth']:shutil.copyfile(ROOT/'out'/n,bundle/n)
 shutil.copyfile(ROOT/'requirements-resolved.txt',bundle/'requirements-resolved.txt')
 shutil.copytree(RUNTIME/'repo/model',bundle/'model',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 shutil.copytree(RUNTIME/'repo/scripts/runpod_sharegpt4v',bundle/'training_source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 hashes={str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()}
 (bundle/'SHA256.json').write_text(json.dumps(hashes,indent=2))
 with tarfile.open(ROOT/'artifacts.tar.gz','w:gz') as t:
  for p in bundle.iterdir():t.add(p,arcname=p.name)
 write('artifact.json',dict(path=str(ROOT/'artifacts.tar.gz'),sha256=sha(ROOT/'artifacts.tar.gz'),files=len(hashes)))

if __name__=='__main__':
 if not (ROOT/'PREPARED').exists():prepare()
 train()
