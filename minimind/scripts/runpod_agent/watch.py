"""Local persistent supervisor: W&B logging, export, HF publication, verified cleanup."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import time
import traceback

from remote import OUT,REMOTE,REPO,ssh,control,credentials

ACTIVE=json.loads((OUT/'active.json').read_text()) if (OUT/'active.json').exists() else {'pod_id':'8ko9ec7m6irjpv','volume_id':'mbuojaqjm8'}
POD=ACTIVE['pod_id']
VOLUME=ACTIVE['volume_id']
MODEL_REPO='caspar/MiniMind-Dense-64M-Agent-RL-v1.0'
PATTERN=re.compile(r'Epoch:\[1/1\]\((\d+)/(\d+)\), Reward:([^,]+), KL:([^,]+), GrpStd:([^,]+), AdvStd:([^,]+), Loss:([^,]+), AvgLen:([^,]+), AdvMean:([^,]+), LR:(\S+)')


def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(name,default=None):
    p=OUT/name;return json.loads(p.read_text()) if p.exists() else default


def poll():
    code='''import json,pathlib
p=pathlib.Path(%r)
s=p/'status.json'
data={'status':json.loads(s.read_text()) if s.exists() else {'stage':'bootstrapping'}}
t=p/'train.log';data['train_log']=t.read_text(errors='replace') if t.exists() else ''
b=p/'bootstrap.log';data['bootstrap_tail']=b.read_text(errors='replace')[-2500:] if b.exists() else ''
print(json.dumps(data))
'''%REMOTE
    r=ssh("python3 - <<'PY'\n"+code+'\nPY',capture_output=True,text=True)
    if r.returncode:raise RuntimeError('SSH poll failed: '+r.stderr[-1000:])
    return json.loads(r.stdout)


def export():
    target=OUT/'results';target.mkdir(exist_ok=True)
    if (target/'completion.json').exists() and read('export_verified.json',{}).get('verified'):return target
    paths=['export','completion.json','prepared.json','preflight.json','reward_preflight.json','evaluation_before.json',
        'evaluation_after.json','training_command.json','smoke_command.json','smoke_complete.json','train.log','smoke.log','out/agent_en_768.pth']
    archive=OUT/'results.tar.gz'
    with archive.open('wb') as f:
        r=ssh('cd '+REMOTE+' && tar -czf - '+' '.join(paths),stdout=f,stderr=-1,timeout=600)
    if r.returncode:raise RuntimeError('Result export failed: '+r.stderr.decode()[-1000:])
    with tarfile.open(archive) as tar:tar.extractall(target,filter='data')
    c=json.loads((target/'completion.json').read_text())
    assert c['complete'] and c['optimizer_updates']==7549 and c['training_rows']==30196
    assert sha(target/'export/model.safetensors')==c['model_sha256']
    assert sha(target/'out/agent_en_768.pth')==c['native_checkpoint_sha256']
    save('export_verified.json',{'model_sha256':c['model_sha256'],'archive_sha256':sha(archive),'verified':True})
    return target


def publish(target,wandb_url):
    from huggingface_hub import HfApi,hf_hub_download
    api=HfApi()
    before=json.loads((target/'evaluation_before.json').read_text())
    after=json.loads((target/'evaluation_after.json').read_text())
    done=json.loads((target/'completion.json').read_text())
    report={'before':before,'after':after,'training':done,'wandb_url':wandb_url}
    export_dir=target/'export'
    (export_dir/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for name in ['prepared.json','preflight.json','reward_preflight.json','completion.json','training_command.json']:
        shutil.copyfile(target/name,export_dir/name)
    (export_dir/'requirements.txt').write_text('torch>=2.8,<3\ntransformers==4.57.1\nsafetensors>=0.4\n')
    card=f'''---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
base_model: caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0
datasets:
- caspar/agent-rl-en
tags:
- minimind
- reinforcement-learning
- agent
- experimental
---

# MiniMind Dense 64M Agent RL v1.0

Full-parameter Agent RL from [caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0](https://huggingface.co/caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0), using the complete **30,196-row** [caspar/agent-rl-en](https://huggingface.co/datasets/caspar/agent-rl-en) dataset. Records mentioning `translate_text` were excluded when that dataset was built.

## Training

The starting policy includes a local 120-update math-tool-call SFT warmup (batch 2, LR 2e-5) derived from the specified base. Before/after evaluation compares against that warmup policy.

One epoch, **7,549 optimizer updates**, batch 4, four rollouts per prompt, CISPO objective, AdamW learning rate 3e-7 with cosine decay to 3e-8, KL coefficient 0.1, BF16, gradient clipping 1.0. SGLang 0.5.5 with the Triton attention backend serves the mathematically equivalent Qwen3 export on one **RTX 5090 32GB**. Native MiniMind performs gradient updates. Policy weights synchronize every 10 batches; checkpoints every 200 batches and at epoch end. Generated responses are limited to 256 tokens per turn and three turns; training retains the final 2,500 tokens of longer trajectories. All training prompts fit within 2,403 tokens; a long trajectory may lose initial context under the training cap.

Reward model for general dialogue: `Skywork/Skywork-Reward-V2-Qwen3-1.7B`, raw score divided by 10 before clipping. Labeled tool tasks use corrected execution and ground-truth rewards: missing calls -1, invalid or failed calls -1.5, and successful execution plus a matching final answer up to +3. Generic reward-model and length rewards cannot offset tool-task failure. Thinking generation is disabled to match the base SFT model, and empty thinking blocks are removed by the saved chat template. Original numeric ground truths are retained. Temperature unit conversion offsets were corrected. General-dialogue formatting and repetition heuristics remain in use.

Training duration (excluding setup/evaluation): **{done['training_seconds']/3600:.2f} hours**. [W&B run]({wandb_url}). Exact pinned source revisions and preflight checks are included in the JSON receipts.

## Sample checks

Same 16 prompts before and after training: 13 tool/ground-truth cases and 3 qualitative English prompts. Greedy generation, maximum 256 tokens per turn, up to three turns.

| Check | Before | After |
| --- | ---: | ---: |
| Ground-truth matches, 13 cases | {before['ground_truth_matches']}/13 | {after['ground_truth_matches']}/13 |
| Expected tool called, 13 cases | {before['expected_tool_called']}/13 | {after['expected_tool_called']}/13 |

Full prompts, responses and tool calls are in `evaluation.json`. This is a small sample check, not a general benchmark. Numeric matches require complete numbers; textual matches remain case-insensitive substring checks. Exact prompt overlap was checked, but paraphrase overlap was not excluded. Machine-translated training data and the reward model may contain errors. The model can repeat itself, miscalculate or fail to call tools; no general reasoning reliability is claimed.

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
repo = "{MODEL_REPO}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True,
    torch_dtype=torch.float32).eval()
text = tok.apply_chat_template([{{"role":"user","content":"Explain why the sky is blue."}}],
    tokenize=False, add_generation_prompt=True, open_thinking=False)
x = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
with torch.inference_mode():
    y = model.generate(inputs=x, max_new_tokens=128, do_sample=False,
        temperature=1.0, top_p=1.0, top_k=0, eos_token_id=tok.eos_token_id)
print(tok.decode(y[0, x.shape[1]:], skip_special_tokens=True))
```

Tool results require an external tool execution loop; the language model does not execute Python or access live weather/time by itself. Training used mock weather, time and exchange-rate data.

## Provenance

MiniMind architecture/code: [jingyaogong/minimind](https://github.com/jingyaogong/minimind), Apache-2.0. The base model's and dataset's provenance/terms continue to apply. The translated dataset retains upstream Apache-2.0 and CC-BY-NC-2.0 notices; this model card does not relicense underlying datasets. See linked base/dataset cards. Only inference weights, tokenizer, architecture code and non-secret evaluation/training receipts are published.
'''
    (export_dir/'README.md').write_text(card)
    if not read('hf_repo_created.json'):
        if api.repo_exists(MODEL_REPO):raise RuntimeError('HF destination already exists; refusing an unreviewed overwrite')
        api.create_repo(MODEL_REPO,private=False)
        save('hf_repo_created.json',{'repo':MODEL_REPO})
    if not read('published.json'):
        commit=api.upload_folder(repo_id=MODEL_REPO,folder_path=export_dir,commit_message='Publish verified one-epoch English Agent RL model')
        path=hf_hub_download(MODEL_REPO,'model.safetensors',revision=commit.oid,local_dir=str(OUT/'published_readback'))
        assert sha(path)==done['model_sha256']
        assert api.model_info(MODEL_REPO).private is False
        save('published.json',{'repo':MODEL_REPO,'url':'https://huggingface.co/'+MODEL_REPO,'commit':commit.oid,'public':True,'sha256_verified':True})
    (OUT/'REPORT.md').write_text(card)
    save('evaluation_summary.json',report)


def remove_task_resources():
    resources=read('resources.json',{'pods':[POD],'volumes':[VOLUME]})
    for resource,identifier in [('pod',p) for p in resources['pods']]:
        name=resource+'_'+identifier+'_deleted.json'
        if read(name):continue
        r=control(resource,'delete',identifier)
        if r.returncode and 'not_found' not in r.stderr:raise RuntimeError(r.stderr)
        r=control(resource,'get',identifier)
        assert r.returncode!=0 and 'not_found' in r.stderr,(r.stdout,r.stderr)
        save(name,{'id':identifier,'confirmed_absent':True,'time':time.time()})


def main():
    import wandb
    os.environ['WANDB_API_KEY']=credentials()['WANDB-API-KEY']
    os.environ['WANDB_CACHE_DIR']=str(OUT/'wandb-cache')
    run=None;cursor=read('wandb_cursor.json',{}).get('step',0);first_failure=None;started=time.time();last_success=time.time()
    while not read('finished.json'):
        try:
            if read('published.json'):
                remove_task_resources();save('finished.json',{'complete':True,'published':read('published.json'),'task_pods_deleted':True,'volumes_preserved':True});return
            payload=poll();last_success=time.time();status=payload['status'];save('live_status.json',status)
            (OUT/'live_train.log').write_text(payload['train_log'])
            stage=status['stage']
            if stage in ['training','final_evaluation','ready_to_publish'] and run is None:
                metadata=read('wandb_run.json',{})
                run=wandb.init(entity='mind4u',project='MiniMind-Agent-RL',id=metadata.get('id'),resume='allow' if metadata else None,
                    name='Dense64M-agent-rl-en-RTX5090-20261026-rewardfix',dir=str(OUT),settings=wandb.Settings(disable_git=True),
                    config={'base_model':'caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0','dataset':'caspar/agent-rl-en','epochs':1,'batch_size':4,'num_generations':4,'max_gen_len':256})
                save('wandb_run.json',{'id':run.id,'url':run.url})
            if run:
                for match in PATTERN.finditer(payload['train_log']):
                    step=int(match[1])
                    if step<=cursor:continue
                    metrics=dict(zip(['reward','kl_ref','group_reward_std','advantages_std','loss','avg_response_len','advantages_mean','learning_rate'],map(float,match.groups()[2:])))
                    run.log(metrics,step=step);cursor=step
                save('wandb_cursor.json',{'step':cursor})
            print(json.dumps(status),flush=True)
            if stage=='ready_to_publish':
                target=export()
                if run:run.summary.update(json.loads((target/'completion.json').read_text()));run.finish();run=None
                publish(target,read('wandb_run.json',{}).get('url',''))
                remove_task_resources();save('finished.json',{'complete':True,'published':read('published.json'),'task_pods_deleted':True,'volumes_preserved':True});return
            if stage=='failed':
                first_failure=first_failure or time.time()
                if time.time()-first_failure>300:
                    control('pod','stop',POD);save('stopped_on_failure.json',status);return
            else:first_failure=None
            if stage=='training' and time.time()-status.get('updated_at',time.time())>1800:
                raise RuntimeError('Training has made no progress for 30 minutes')
            if stage=='bootstrapping' and time.time()-started>2400:
                raise RuntimeError('Bootstrap exceeded 40 minutes')
        except Exception as exc:
            save('supervisor_error.json',{'error':str(exc),'traceback':traceback.format_exc(),'time':time.time()})
            print(traceback.format_exc(),flush=True)
            if time.time()-last_success>1800:
                control('pod','stop',POD);save('stopped_on_failure.json',{'reason':'Cannot monitor task Pod for 30 minutes'});return
        time.sleep(30)


if __name__=='__main__':main()
