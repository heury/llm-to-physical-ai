"""Capture and verify artifacts, publish model/data, then remove only the task Pod."""
import hashlib,json,os,shlex,shutil,subprocess,sys,tarfile,time
from pathlib import Path
os.environ.setdefault('HF_XET_HIGH_PERFORMANCE','1')
import requests
from huggingface_hub import HfApi,hf_hub_download,snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError
ROOT=Path('/home/mind4u/dev/minimind/outputs/minimind/dpo-short-20261025');REMOTE='/shared-volume/minimind-dense-dpo-short-20261025'
REPOS={'model':'caspar/MiniMind-Dense-64M-UltraFeedback-DPO-3K-v1.0','dataset':'caspar/MiniMind-UltraFeedback-Short340-DPO-3K-v1.0'}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def log(phase,**kw):
 obj=dict(phase=phase,timestamp=time.time(),**kw);write(ROOT/'status.json',obj);print(json.dumps(obj),flush=True)
def ssh(command,**kw):
 p=json.loads((ROOT/'pod.json').read_text())['ssh']['direct'];return subprocess.run(['ssh','-i',str(Path.home()/'.runpod/ssh/runpodctl-ssh-key'),'-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=30','-p',str(p['port']),p['username']+'@'+p['host'],command],**kw)
def read(command):return ssh(command,capture_output=True,text=True,check=True).stdout

def capture():
 if (ROOT/'capture-verified.json').exists():return
 log('waiting_for_training');deadline=time.time()+3600
 while read(f'test -f {REMOTE}/results/completion.json && echo complete || echo waiting').strip()!='complete':
  tail=read(f'tail -10 {REMOTE}/train.log')
  if 'Traceback' in tail or read(f'test -f {REMOTE}/results/failed.json && echo failed || echo ok').strip()=='failed':raise RuntimeError(tail)
  if time.time()>deadline:raise TimeoutError('Training did not finish')
  time.sleep(15)
 log('capture_artifacts')
 code="""import hashlib,json,tarfile
from pathlib import Path
r=Path('/shared-volume/minimind-dense-dpo-short-20261025')
files=[p for folder in ['results','model'] for p in (r/folder).rglob('*') if p.is_file() and p.name!='best.pth']
m={}
for p in files:
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 m[str(p.relative_to(r))]={'bytes':p.stat().st_size,'sha256':digest}
(r/'artifact-manifest.json').write_text(json.dumps(m,indent=2))
with tarfile.open('/tmp/dpo-artifacts.tar.gz','w:gz') as t:
 for p in files+[r/'artifact-manifest.json']:t.add(p,arcname=str(p.relative_to(r)))
print('ready')
"""
 read('python -c '+shlex.quote(code))
 with (ROOT/'artifacts.tar.gz').open('wb') as f:ssh('cat /tmp/dpo-artifacts.tar.gz',stdout=f,check=True)
 dest=ROOT/'artifacts';dest.mkdir(exist_ok=True)
 with tarfile.open(ROOT/'artifacts.tar.gz') as t:t.extractall(dest,filter='data')
 m=json.loads((dest/'artifact-manifest.json').read_text())
 for name,info in m.items():assert sha(dest/name)==info['sha256'] and (dest/name).stat().st_size==info['bytes'],name
 write(ROOT/'capture-verified.json',{'files':len(m),'verified':True})

def cards():
 done=json.loads((ROOT/'artifacts/results/completion.json').read_text());manifest=json.loads((ROOT/'dataset_manifest.json').read_text());assert done['complete'] and done['reload_weights_and_scores_exact'] and done['reference_unchanged']
 stage=ROOT/'release';stage.mkdir(exist_ok=True);model=stage/'model';data=stage/'dataset'
 shutil.copytree(ROOT/'artifacts/model',model,dirs_exist_ok=True);data.mkdir(exist_ok=True)
 for n in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja']:
  if (ROOT/'base'/n).exists():shutil.copyfile(ROOT/'base'/n,data/n)
 shutil.copytree(ROOT/'data',data/'data',dirs_exist_ok=True)
 for folder in [data,model]:
  shutil.copyfile(ROOT/'dataset_manifest.json',folder/'dataset_manifest.json');shutil.copyfile(ROOT/'data_checks.json',folder/'data_checks.json')
  (folder/'recipe').mkdir(exist_ok=True)
  for p in Path(__file__).parent.glob('*.py'):
   if p.name in ['prepare.py','check_data.py','train.py','recover_export.py']:shutil.copyfile(p,folder/'recipe'/p.name)
 shutil.copyfile(ROOT/'base/LICENSE',model/'LICENSE');shutil.copyfile(ROOT/'dataset-README.md',data/'UPSTREAM_README.md')
 for n in ['completion.json','before.json','after.json','validation_history.json','samples.json','smoke.json','settings.json','environment.json','metrics.jsonl']:
  shutil.copyfile(ROOT/'artifacts/results'/n,model/n)
 (model/'requirements.txt').write_text('torch>=2.8,<3\ntransformers==4.57.6\nhuggingface_hub>=0.36\nsafetensors>=0.4\n')
 data_card=f'''---
license: mit
language:
- en
task_categories:
- text-generation
tags:
- dpo
- preference
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
  - split: validation
    path: data/validation.parquet
  - split: test
    path: data/test.parquet
---
# MiniMind UltraFeedback Short340 DPO 3K — v1.0

Exactly 3,000 unique-prompt pairs selected from [{manifest['source']['repo']}](https://huggingface.co/datasets/{manifest['source']['repo']}/tree/{manifest['source']['revision']}). Source revision `{manifest['source']['revision']}`; {manifest['source_rows']:,} scanned rows, {manifest['unique_eligible_prompts']:,} eligible unique prompts. Attribution: Argilla / Notus and the original OpenBMB UltraFeedback creators; MIT per the upstream dataset card, preserved in UPSTREAM_README.md.

**2,700 train / 150 validation / 150 test.** The source only has a train split; these are newly created heldouts. Canonical prompt groups cannot cross splits. Seed 42 hash selection across all eligible rows; original chosen/rejected texts, scores, source and source row indices are retained. Empty responses and identical tokenized pairs are excluded. Selected rating ties: zero.

Both **full prompt + completion + chat markers + EOS** must be **strictly <340 tokens**, using the bundled 6,400-token tokenizer of `{manifest['base']['repo']}` at `{manifest['base']['revision']}`. There is **no truncation**. Empty thinking blocks are deterministically removed to match base SFT. No extra system message is added beyond the original tokenizer template. Only the final assistant completion including EOS/newline is labeled; prompt and padding have -100 labels.

Parquet/JSONL retain readable examples. NPY arrays have shape `[pairs,2,2,340]`, int16: second axis chosen/rejected; third axis input_ids/labels. Padding is added only after filtering. Non-padding maximum is 339. See dataset_manifest.json and data_checks.json for exact counts, hashes and lengths.

DPO splits are disjoint; exposure during prior SFT/pretraining is not excluded. Short-response filtering changes the task/source distribution. This is not a general benchmark.

```python
from datasets import load_dataset
data = load_dataset("{REPOS['dataset']}", revision="v1.0")
print(data["train"][0]["chosen"])
```

Training model: [{REPOS['model']}](https://huggingface.co/{REPOS['model']}). Preparation and training recipes are included as reference; their experiment paths need adapting for other runs. SHA256SUMS.json covers all packaged files except itself.
'''
 (data/'README.md').write_text(data_card)
 b=done['before']['test'];a=done['after']['test'];settings=json.loads((ROOT/'artifacts/results/settings.json').read_text())
 model_card=f'''---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
base_model: {manifest['base']['repo']}
datasets:
- {REPOS['dataset']}
tags:
- minimind
- dpo
- experimental
---
# MiniMind Dense 64M UltraFeedback DPO 3K — v1.0

Full-parameter DPO of [{manifest['base']['repo']}](https://huggingface.co/{manifest['base']['repo']}/tree/{manifest['base']['revision']}). Base revision `{manifest['base']['revision']}`. Architecture code and base model license: Apache-2.0, derived from jingyaogong/minimind. Upstream training-data licenses still apply.

63,912,192 parameters, 768 hidden size, 8 layers, 6,400 vocabulary. **FP32 safetensors** preserve small DPO updates; BF16 autocast was used for training and reported GPU inference. The base served as a frozen reference, checked unchanged after training.

## Training

Dataset: [{REPOS['dataset']}](https://huggingface.co/datasets/{REPOS['dataset']}); 3,000 selected pairs, 2,700 train /150 validation /150 test. Full chosen and rejected sequences each <340 tokens, no truncation. Prompt/padding masked; final-assistant log probabilities summed. Original base chat template with empty thinking block removed.

Two epochs, {done['optimizer_steps']} optimizer updates; batch32 preference pairs; AdamW LR5e-7, beta0.1, warmup10%, cosine decay to10%, weight decay0, clip1.0, seed42. Validation DPO loss selected epoch **{done['selected_epoch']}**; test data was not used for selection. Elapsed through the final optimizer update (including inter-epoch validation): {done['training_seconds']:.1f}s on one RTX5090. See settings.json, environment.json and metrics.jsonl.

## Heldout evaluation (150 pairs)

| Metric | Base SFT | DPO |
|---|---:|---:|
| DPO loss | {b['dpo_loss']:.6f} | {a['dpo_loss']:.6f} |
| Raw sequence preference accuracy | {100*b['raw_sequence_preference_accuracy']:.2f}% | {100*a['raw_sequence_preference_accuracy']:.2f}% |
| Per-token preference accuracy | {100*b['per_token_preference_accuracy']:.2f}% | {100*a['per_token_preference_accuracy']:.2f}% |
| Chosen response NLL | {b['chosen_nll']:.6f} | {a['chosen_nll']:.6f} |
| Chosen response PPL | {b['chosen_ppl']:.4f} | {a['chosen_ppl']:.4f} |

Reference-relative reward accuracy after DPO: {100*a['reference_relative_accuracy']:.2f}%. The identical baseline is tied by definition: this is **not a baseline semantic win rate**. Raw sequence preference accuracy is length-sensitive. A small DPO heldout set and likelihood metrics do not establish better general reasoning or factual reliability. Prior SFT/pretraining exposure has not been excluded. Generated answers can still repeat or be incorrect.

`samples.json` includes every one of 16 fixed before/after greedy comparisons: eight general questions and the first eight heldout prompts, not cherry-picked successes. Maximum128 new tokens, total context <=340, no repetition penalty. `completion.json` records unchanged reference, changed policy parameters and exact exported-model reload checks.

## Usage

Review the bundled custom model code before enabling trust_remote_code.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "{REPOS['model']}"
tok = AutoTokenizer.from_pretrained(repo, revision="v1.0")
model = AutoModelForCausalLM.from_pretrained(repo, revision="v1.0", trust_remote_code=True, torch_dtype=torch.float32).eval()
messages = [{{"role":"user", "content":"Why is the sky blue?"}}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=False)
text = text.replace("<think>\\n\\n</think>\\n\\n", "")
x = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
out = model.generate(x, max_new_tokens=min(128, 340-x.shape[1]), do_sample=False, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0)
print(tok.decode(out[0,x.shape[1]:], skip_special_tokens=True))
```

CPU FP32 output can differ from reported CUDA BF16 greedy outputs. Training used context340; larger architectural position buffers do not establish longer-context quality. SHA256SUMS.json covers packaged files except itself.
'''
 (model/'README.md').write_text(model_card)
 for kind,folder in [('dataset',data),('model',model)]:
  manifest_files={str(p.relative_to(folder)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json' and '.cache' not in p.parts}
  write(folder/'SHA256SUMS.json',manifest_files)
 return stage

def publish(stage):
 api=HfApi();assert api.whoami()['name']=='caspar';public=HfApi(token=False)
 for kind in ['dataset','model']:
  receipt=ROOT/f'{kind}-published.json'
  if receipt.exists():continue
  repo=REPOS[kind];folder=stage/kind;owned=ROOT/f'{kind}-repo-created.json'
  if not owned.exists():
   try:api.repo_info(repo,repo_type=kind)
   except RepositoryNotFoundError:pass
   else:raise RuntimeError('Refusing to overwrite existing unowned '+repo)
   api.create_repo(repo,repo_type=kind,private=False);write(owned,{'repo':repo})
  log('upload',kind=kind,repo=repo);result=api.upload_folder(repo_id=repo,repo_type=kind,folder_path=folder,commit_message='Release verified short-context DPO v1.0',ignore_patterns=['.cache/**']);info=public.repo_info(repo,repo_type=kind,revision=result.oid,files_metadata=True);assert not info.private
  manifest=json.loads((folder/'SHA256SUMS.json').read_text());manifest['SHA256SUMS.json']={'bytes':(folder/'SHA256SUMS.json').stat().st_size,'sha256':sha(folder/'SHA256SUMS.json')};entries={x.rfilename:x for x in info.siblings}
  for name,expected in manifest.items():
   ent=entries[name];assert ent.size==expected['bytes'],name
   if ent.lfs:assert ent.lfs.sha256==expected['sha256'],name
   else:
    p=hf_hub_download(repo,name,repo_type=kind,revision=result.oid,token=False,cache_dir=ROOT/'verification-cache');assert sha(p)==expected['sha256'],name
  api.create_tag(repo,tag='v1.0',revision=result.oid,repo_type=kind,exist_ok=True)
  assert public.repo_info(repo,repo_type=kind,revision='v1.0').sha==result.oid
  write(receipt,{'repo':repo,'tag':'v1.0','commit':result.oid,'private':False,'verified_files':len(manifest),'verified_at':time.time()})

def verify_loading():
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from datasets import load_dataset
 log('verify_public_loading');torch.set_num_threads(4)
 data=load_dataset(REPOS['dataset'],revision='v1.0',token=False)
 assert {s:len(data[s]) for s in data}=={'train':2700,'validation':150,'test':150}
 p=Path(snapshot_download(REPOS['model'],revision='v1.0',token=False,local_dir=ROOT/'public-model'))
 done=json.loads((ROOT/'artifacts/results/completion.json').read_text());assert sha(p/'model.safetensors')==done['model_sha256']
 model=AutoModelForCausalLM.from_pretrained(p,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32).eval();tok=AutoTokenizer.from_pretrained(p,local_files_only=True)
 text=tok.apply_chat_template([{'role':'user','content':'What is the capital of France?'}],tokenize=False,add_generation_prompt=True,open_thinking=False).replace('<think>\n\n</think>\n\n','');x=tok(text,add_special_tokens=False,return_tensors='pt').input_ids
 with torch.inference_mode():out=model.generate(x,max_new_tokens=32,do_sample=False,temperature=1.,top_p=1.,top_k=0)
 write(ROOT/'public-loading-verified.json',{'dataset_rows':{s:len(data[s]) for s in data},'public_model_sha256':sha(p/'model.safetensors'),'parameters':sum(p.numel() for p in model.parameters()),'cpu_smoke_output':tok.decode(out[0,x.shape[1]:],skip_special_tokens=True)})

def cleanup():
 config={}
 for line in Path('/home/mind4u/dev/minimind/.env').read_text().splitlines():
  if '=' in line and not line.lstrip().startswith('#'):
   k,v=line.split('=',1);config[k.strip()]=v.strip().strip('\"\'')
 session=requests.Session();session.headers['Authorization']='Bearer '+config['RUNPOD-API-KEY'];pod=json.loads((ROOT/'pod.json').read_text());assert pod['id']=='f6078textxf5yr'
 url='https://api.runpod.io/v2/pods/'+pod['id'];r=session.get(url,timeout=45);r.raise_for_status();assert r.json()['name']=='minimind-dense-dpo-short-20261025';r=session.delete(url,timeout=45);r.raise_for_status();r=session.get(url,timeout=45);assert r.status_code==404
 write(ROOT/'cleanup.json',{'pod_id':pod['id'],'deleted':True,'readback_status':404,'timestamp':time.time(),'global_volume_preserved':True})

def main():
 capture();stage=cards()
 if not (ROOT/'wandb.json').exists():subprocess.run(['/tmp/minimind-sft-tools/bin/python',str(Path(__file__).with_name('log_wandb.py'))],check=True)
 publish(stage);verify_loading();cleanup();log('complete',repos=REPOS,pod_deleted=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:log('failed',error=repr(e));raise
