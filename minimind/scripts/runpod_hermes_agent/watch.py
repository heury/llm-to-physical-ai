"""Local-only credentials: publish both stages, verify bytes, then remove task Pods."""
import hashlib,io,json,os,re,shutil,tarfile,time,traceback,shlex
from pathlib import Path
from remote import OUT,REMOTE,ssh,control,credentials
SFT_REPO='caspar/MiniMind-Dense-64M-Hermes-FunctionCalling-SFT-v1.0'
RL_REPO='caspar/MiniMind-Dense-64M-Hermes-Agent-RL-v1.0'

def save(name,data):
    p=OUT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def read(name,default=None):
    p=OUT/name;return json.loads(p.read_text()) if p.exists() else default
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def poll():
    code="import json,pathlib;p=pathlib.Path(%r);s=p/'status.json';print(json.dumps({'status':json.loads(s.read_text()) if s.exists() else {'stage':'bootstrapping'},'train_log':(p/'train.log').read_text() if (p/'train.log').exists() else ''}))"%REMOTE
    r=ssh('python3 -c '+shlex.quote(code),capture_output=True,text=True);assert r.returncode==0,r.stderr
    return json.loads(r.stdout)
def fetch(stage):
    target=OUT/stage;target.mkdir(exist_ok=True)
    if read(stage+'_export_verified.json'):return target
    names=['prepared.json','sft_manifest.json','sft_before.json','sft_after.json','sft_completion.json']
    if stage=='sft':names+=['export_sft']
    else:names+=['export','completion.json','reward_preflight.json','preflight.json','evaluation_before.json','evaluation_after.json','rl_quality.json','training_command.json','train.log','sft_published.json']
    r=ssh('cd '+REMOTE+' && tar -czf - '+' '.join(names),capture_output=True,timeout=180)
    assert r.returncode==0,r.stderr.decode()[-1000:]
    with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as tar:tar.extractall(target,filter='data')
    folder=target/('export_sft' if stage=='sft' else 'export')
    done=json.loads((target/('sft_completion.json' if stage=='sft' else 'completion.json')).read_text())
    assert done['complete'] and digest(folder/'model.safetensors')==done['model_sha256']
    if stage=='rl':assert done['optimizer_updates']==7549 and done['training_rows']==30196
    save(stage+'_export_verified.json',{'sha256':done['model_sha256'],'verified':True})
    return target

def publish(stage):
    from huggingface_hub import HfApi,hf_hub_url
    import requests
    receipt=read(stage+'_published.json')
    if receipt:return receipt
    repo=SFT_REPO if stage=='sft' else RL_REPO;target=fetch(stage);folder=target/('export_sft' if stage=='sft' else 'export')
    results={f.stem:json.loads(f.read_text()) for f in target.glob('*.json')}
    for f in target.glob('*.json'):shutil.copyfile(f,folder/f.name)
    manifest=results['sft_manifest'];done=results['sft_completion' if stage=='sft' else 'completion']
    if stage=='sft':
        summary=f"Hermes SFT: {manifest['train_conversations']} training conversations, {manifest['validation_conversations']} validation conversations; {manifest['train_examples']} retained assistant targets; 3 epochs / {done['optimizer_updates']} updates. Validation CE: {done['before_validation_loss']:.4f} → {done['after_validation_loss']:.4f}. Best epoch selected by validation CE."
    else:
        before=results['evaluation_before'];after=results['evaluation_after']
        summary=f"Agent RL: all 30,196 agent-rl-en rows, 1 epoch / 7,549 updates. Small 13-case tool sample check: final-answer matches {before['ground_truth_matches']}/13 → {after['ground_truth_matches']}/13; expected tool name {before['expected_tool_called']}/13 → {after['expected_tool_called']}/13. Quality improvement: {results['rl_quality']['improved']}. This experimental checkpoint is published with its actual results, not a claim of improved task performance."
    base='caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0' if stage=='sft' else SFT_REPO
    card=f'''---
language:
- en
library_name: transformers
pipeline_tag: text-generation
base_model: {base}
datasets:
- NousResearch/hermes-function-calling-v1
'''+('- caspar/agent-rl-en\n' if stage=='rl' else '')+f'''tags:
- minimind
- function-calling
- experimental
---

# {repo.split('/')[-1]}

{summary}

## Training and provenance

Base architecture: MiniMind Dense 64M. Original base revision: `7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d`. Hermes source: **func-calling.json only**, revision `{manifest['revision']}`. Source rows: {manifest['source_rows']}. Conversion re-renders tools once in the existing MiniMind chat template; assistant calls and post-tool answers are supervised; system/user/tool/padding labels are masked. Conversation groups are split by a deterministic hash of the user text. Overlength assistant targets skipped without truncating their endings: {manifest['overlength_examples_skipped']}. Invalid rows excluded: {len(manifest['rejected'])}. All exclusions are recorded. No synthesized completion data was added.

SFT: full parameters, BF16 compute with FP32 training weights, batch 1 / accumulation 8, LR 2e-5 cosine to 2e-6, 3 epochs, max length 4096, seed 42, gradient clip 1.0. Epoch checkpoint chosen by validation CE. Validation generation examples are for inspection, not a held-out benchmark; function templates may overlap between splits.

Agent RL (RL artifact only): `trainer/train_agent.py`, CISPO, batch 4, four generations per prompt, response microbatch 1, LR 3e-7 cosine to 3e-8, beta 0.1, max 3 turns / 256 tokens per turn / 2500 retained training tokens, policy synchronization every 10 steps. SGLang 0.5.5 serves a weight-compatible Qwen3 export; the published model is native MiniMind. Reward model for general dialogue: Skywork/Skywork-Reward-V2-Qwen3-1.7B, revision e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc, raw scores divided by 10 then clipped. Labeled tool tasks use execution/answer matching; unfinished trajectories are penalized -2 and repeated identical successful calls are penalized. Mock tools do not access live weather/time/exchange rates.

Both stages use one RTX 5090. Weights were strict-reloaded and checked finite. Exact settings, sample generations and raw metrics are in the adjacent JSON files. General capability, factual accuracy and reliable tool use are not guaranteed. Repeated calls or unsupported tool names may still occur. The 13-case agent sample is small and is not a general benchmark. Numeric matches use complete numbers; text matches remain case-insensitive substrings.

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
repo = '{repo}'
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.float32).eval()
text = tok.apply_chat_template([{{'role':'user','content':'Hello'}}], tokenize=False, add_generation_prompt=True, open_thinking=False)
x = tok(text, return_tensors='pt', add_special_tokens=False).input_ids
with torch.no_grad():
    y = model.generate(x, max_new_tokens=128, do_sample=False, eos_token_id=tok.eos_token_id)
print(tok.decode(y[0, x.shape[1]:], skip_special_tokens=True))
```

Tool execution requires an external loop. Retain the tool schema and tool-response roles used in training.

Hermes is distributed with an Apache-2.0 notice. The original MiniMind code/base and other source terms continue to apply. The translated Agent RL dataset retains upstream Apache-2.0 and CC-BY-NC-2.0 notices; this card does not relicense those sources. See the source dataset/model cards.
'''
    (folder/'README.md').write_text(card);(folder/'requirements.txt').write_text('torch>=2.8,<3\ntransformers==4.57.1\nsafetensors>=0.4\n')
    api=HfApi();created=read(stage+'_repo_created.json')
    if not created:
        if api.repo_exists(repo):raise RuntimeError('Destination exists without this task ownership: '+repo)
        api.create_repo(repo,private=False);save(stage+'_repo_created.json',{'repo':repo})
    commit=api.upload_folder(repo_id=repo,folder_path=folder,commit_message='Publish verified '+stage+' experimental checkpoint and evaluations')
    h=hashlib.sha256()
    with requests.get(hf_hub_url(repo,'model.safetensors',revision=commit.oid),stream=True,timeout=120) as response:
        response.raise_for_status()
        for chunk in response.iter_content(1024*1024):h.update(chunk)
    assert h.hexdigest()==done['model_sha256'] and api.model_info(repo).private is False
    receipt={'repo':repo,'url':'https://huggingface.co/'+repo,'commit':commit.oid,'public':True,'sha256_verified':True}
    save(stage+'_published.json',receipt);(OUT/(stage+'_REPORT.md')).write_text(card)
    return receipt

def acknowledge_sft(receipt):
    code="import json,pathlib;p=pathlib.Path(%r);p.write_text(%r)"%(REMOTE+'/sft_published.json',json.dumps(receipt))
    r=ssh('python3 -c '+shlex.quote(code),capture_output=True,text=True);assert r.returncode==0,r.stderr

def cleanup():
    for pod in read('resources.json')['pods']:
        name=pod+'_deleted.json'
        if read(name):continue
        r=control('pod','delete',pod)
        if r.returncode and 'not_found' not in r.stderr:raise RuntimeError(r.stderr)
        r=control('pod','get',pod);assert r.returncode and 'not_found' in r.stderr
        save(name,{'id':pod,'confirmed_absent':True})

def main():
    started=time.time();last_success=started;failed=None;run=None;cursor=0;last_sft=0;error_started=None
    os.environ['WANDB_API_KEY']=credentials()['WANDB-API-KEY']
    while not read('finished.json'):
        try:
            if read('rl_published.json'):
                cleanup();save('finished.json',{'complete':True,'sft':read('sft_published.json'),'rl':read('rl_published.json'),'task_pods_deleted':True,'volumes_preserved':True});return
            payload=poll();status=payload['status'];stage=status['stage'];last_success=time.time()
            save('live_status.json',status);(OUT/'live_train.log').write_text(payload['train_log'])
            print(json.dumps(status),flush=True)
            if stage in ('sft_training','training'):
                import wandb
                if run is None:
                    old=read('wandb_run.json',{})
                    run=wandb.init(entity='mind4u',project='MiniMind-Hermes-Agent',id=old.get('id'),resume='allow' if old else None,name='Dense64M-Hermes-SFT-Agent-20261026',dir=str(OUT),settings=wandb.Settings(disable_git=True))
                    save('wandb_run.json',{'id':run.id,'url':run.url})
                if stage=='sft_training' and status['step']>last_sft:
                    run.log({'sft/step':status['step'],'sft/loss':status['loss']});last_sft=status['step']
                if stage=='training':
                    for line in payload['train_log'].splitlines():
                        match=re.search(r'Epoch:\[1/1\]\((\d+)/(\d+)\)',line)
                        if match and int(match[1])>cursor:
                            metrics={k:float(v) for k,v in re.findall(r'(Reward|KL|GrpStd|AdvStd|Loss|AvgLen|LR):([-+\d.e]+)',line)}
                            cursor=int(match[1]);run.log({'rl/step':cursor,**{'rl/'+k:v for k,v in metrics.items()}})
            if stage=='sft_ready_to_publish':acknowledge_sft(publish('sft'))
            if stage=='ready_to_publish':
                publish('rl')
                if run:run.finish();run=None
                cleanup();save('finished.json',{'complete':True,'sft':read('sft_published.json'),'rl':read('rl_published.json'),'task_pods_deleted':True,'volumes_preserved':True});return
            if stage=='failed':
                failed=failed or time.time()
                if time.time()-failed>300:
                    control('pod','stop',read('active.json')['pod_id']);save('stopped_on_failure.json',status);return
            else:failed=None
            if stage in ('sft_training','training') and time.time()-status.get('updated_at',time.time())>1800:raise RuntimeError('No optimizer progress for 30 minutes')
            if stage=='bootstrapping' and time.time()-started>2400:raise RuntimeError('Bootstrap exceeded 40 minutes')
            error_started=None
        except Exception as exc:
            error_started=error_started or time.time()
            save('supervisor_error.json',{'error':str(exc),'traceback':traceback.format_exc(),'time':time.time()})
            print(traceback.format_exc(),flush=True)
            if time.time()-last_success>1800 or time.time()-error_started>1200:
                control('pod','stop',read('active.json')['pod_id']);save('stopped_on_failure.json',{'error':str(exc)});return
        time.sleep(30)

if __name__=='__main__':main()
