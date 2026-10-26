"""Export verified artifacts, remove task GPU, publish and verify immutable HF commit."""
import hashlib,json,shutil,sys,tarfile,time
from pathlib import Path
from remote import OUT,ssh,cli
from huggingface_hub import HfApi,hf_hub_download
ROOT='/workspace/moe-dense-gsm8k-distill-20261025'
REPO_ID='caspar/MiniMind-Dense-64M-GSM8K-SoftDistill-MoE-v1.0'
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 r=ssh('cat '+ROOT+'/completion.json',capture_output=True,text=True);r.check_returncode();c=json.loads(r.stdout)
 assert c['complete'] and c['epochs']==3 and c['updates']==345 and c['checkpoint_reload_verified']
 assert c['baseline']['test_rows']==c['distilled']['test_rows']==1319
 with (OUT/'result.tar').open('wb') as f:
  ssh('tar -C '+ROOT+' -cf - model artifact_manifest.json completion.json training.log environment.txt code',stdout=f).check_returncode()
 dest=OUT/'result';dest.mkdir(exist_ok=True)
 with tarfile.open(OUT/'result.tar') as t:t.extractall(dest,filter='data')
 manifest=json.loads((dest/'artifact_manifest.json').read_text())
 for name,want in manifest.items():assert digest(dest/name)==want,name
 (OUT/'export_verified.json').write_text(json.dumps(dict(files=len(manifest),sha256=digest(OUT/'result.tar'))))
 p=json.loads((OUT/'pod.json').read_text());cli('pod','delete',p['id'])
 assert p['id'] not in [v['id'] for v in cli('pod','list')]
 (OUT/'cleanup.json').write_text(json.dumps(dict(pod_id=p['id'],deleted=True,existing_volume_preserved=True)))
 folder=dest/'model';s=json.loads((folder/'sources.json').read_text());data=json.loads((folder/'data.json').read_text())
 b,a=c['baseline'],c['distilled']
 card=f'''---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
base_model: {s['student']['repo']}
tags:
- minimind
- knowledge-distillation
- gsm8k
- experimental
datasets:
- caspar/MiniMind-Mixed-SFT-Training-v1.0
metrics:
- accuracy
---
# MiniMind Dense 64M GSM8K soft distillation

Full-parameter soft knowledge distillation from `{s['teacher']['repo']}` into
`{s['student']['repo']}`. Exact source revisions are in `sources.json`.
This is an experimental small model; the reported metrics do not imply reliable mathematical reasoning.

## Training

Use only the released GSM8K training subset: {data['train_rows']} conversations,
{data['train_windows']} windows, {data['train_targets']} assistant targets per epoch.
Identical 6400-token tokenizers. Frozen teacher, full-softmax forward KL:
`loss = 0.5 * CE + 0.5 * T^2 * KL(teacher || student)`, temperature 1.5.
Assistant-only shifted targets; prompts/padding masked. 340-token windows,
stride 256, overlap targets counted once, final partial windows retained.
Continuation windows may not contain the full question.
Three epochs / 345 optimizer updates; microbatch 16 x accumulation 4,
normalized by actual target-token counts. AdamW, LR cosine 5e-6 to 5e-7,
weight decay .01, gradient clip 1, seed 42. BF16 computation, FP32 optimizer
weights, exported FP16 weights. A40 GPU. See `settings.json` and `metrics.jsonl`.

## GSM8K test evaluation

All 1319 official main/test examples, released as `validation_gsm8k`, excluded
from training and checkpoint selection. Fixed final epoch, no test tuning.
Zero-shot greedy, max 256 new tokens and total context 340; exact-length batches
without padding. Overflow/invalid outputs remain in the denominator.
Prompt suffix: `Solve step by step. End your answer with #### followed by the final number.`
Empty think blocks are removed in training and inference.

| Model | Strict #### answer | Last-number fallback |
|---|---:|---:|
| Original Dense student | {b['strict_correct']}/1319 ({b['strict_accuracy']:.2%}) | {b['flexible_correct']}/1319 ({b['flexible_accuracy']:.2%}) |
| Distilled Dense | {a['strict_correct']}/1319 ({a['strict_accuracy']:.2%}) | {a['flexible_correct']}/1319 ({a['flexible_accuracy']:.2%}) |

Assistant-only test CE: {c['validation_before']['ce']:.6f} -> {c['validation_after']['ce']:.6f}.
Teacher KL at T=1.5: {c['validation_before']['kl']:.6f} -> {c['validation_after']['kl']:.6f}.
All generations and evaluation settings are included. Scores use the reloaded
published FP16 checkpoint, not in-memory training weights. No claim of
paraphrase-level or upstream-pretraining decontamination.

## Loading

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "{REPO_ID}"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(
    repo, trust_remote_code=True, torch_dtype=torch.float32,
).to("cuda").eval()
messages = [{{"role": "user", "content": "What is 7 times 8?\\nSolve step by step. End your answer with #### followed by the final number."}}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=False)
text = text.replace("<think>\\n\\n</think>\\n\\n", "")
x = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    y = model.generate(inputs=x, max_new_tokens=min(256, 340-x.shape[1]), do_sample=False, top_k=0, top_p=1., temperature=1.)
print(tokenizer.decode(y[0, x.shape[1]:], skip_special_tokens=True))
```

Use `transformers==4.57.6` and `torch>=2.6,<3`. Custom MiniMind modeling code is
included. Base models/code are Apache-2.0; GSM8K data retains MIT terms.
'''
 (folder/'README.md').write_text(card)
 (folder/'requirements.txt').write_text('torch>=2.6,<3\ntransformers==4.57.6\nsafetensors>=0.4\n')
 lic=hf_hub_download(s['student']['repo'],'LICENSE',revision=s['student']['revision']);shutil.copyfile(lic,folder/'LICENSE')
 shutil.copytree(dest/'code',folder/'recipe',dirs_exist_ok=True)
 shutil.copyfile(dest/'environment.txt',folder/'environment.txt')
 # Local reload is an additional check before any public upload.
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 m=AutoModelForCausalLM.from_pretrained(folder,trust_remote_code=True,torch_dtype=torch.float32).eval()
 with torch.inference_mode():assert torch.isfinite(m(torch.tensor([[1,20,21,2]])).logits).all()
 api=HfApi();assert api.whoami()['name']=='caspar'
 api.create_repo(REPO_ID,private=False,exist_ok=False)
 commit=api.upload_folder(repo_id=REPO_ID,folder_path=folder,commit_message='Publish verified GSM8K MoE-to-Dense soft distillation and full test evaluation')
 local=digest(folder/'model.safetensors')
 remote=hf_hub_download(REPO_ID,'model.safetensors',revision=commit.oid,local_dir=OUT/'published-verification')
 assert digest(remote)==local
 (OUT/'published.json').write_text(json.dumps(dict(repo=REPO_ID,url='https://huggingface.co/'+REPO_ID,commit=commit.oid,model_sha256=local,upload_verified=True),indent=2))
 print('PUBLISHED','https://huggingface.co/'+REPO_ID,commit.oid,flush=True)
if __name__=='__main__':main()
