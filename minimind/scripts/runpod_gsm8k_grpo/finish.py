"""Local supervisor: export verified artifacts, release task GPU, publish models.

No API token is sent to the training Pod. Can be restarted idempotently.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

from huggingface_hub import HfApi, hf_hub_download
from control import credentials
from remote import ssh, ROOT, OUT, PROJECT

POD=json.loads((OUT/'pod.json').read_text())['id']

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def cli(*args):
    env=dict(os.environ,RUNPOD_API_KEY=credentials()['RUNPOD-API-KEY'])
    return subprocess.run([str(Path.home()/'.local/bin/runpodctl'),*args],env=env,capture_output=True,text=True,timeout=60)

def export(kinds):
    target=OUT/'results';target.mkdir(exist_ok=True)
    names=['settings.json','dataset_manifest.json','status.json','all_complete.json','bootstrap.log']
    if len(kinds) > 1:
        names += ['cross_reward_scores.json','bootstrap-moe-continuation.log']
    names += [kind+'-'+label for kind in kinds for label in ['baseline','rule','skywork']]
    archive=OUT/'results.tar.gz'
    with archive.open('wb') as f:
        result=ssh('tar --exclude="checkpoint_*.pth" --exclude="checkpoint_*.pth.tmp" -czf - -C '+ROOT+' '+' '.join(map(shlex.quote,names)),
                   stdout=f,stderr=subprocess.PIPE,timeout=600)
    result.check_returncode()
    with tarfile.open(archive) as tar:tar.extractall(target,filter='data')
    for kind in kinds:
        baseline=read(target/(kind+'-baseline/evaluation.json'))
        assert len(baseline['rows'])==1319
        for mode in ['rule','skywork']:
            folder=target/(kind+'-'+mode);done=read(folder/'completion.json')
            assert done['complete'] and done['checkpoint_verified'] and done['package_verified']
            assert done['epochs']==1 and done['train_rows']==6851 and done['optimizer_updates']==215
            assert len(read(folder/'evaluation_after.json')['rows'])==1319
            assert all(sha(folder/'model'/name)==digest for name,digest in done['model_files'].items())
            rows=[json.loads(line) for line in (folder/'metrics.jsonl').open()]
            assert rows[-1]['step']==215
            batches=read(folder/'batch_indices.json')
            indices=[i for batch in batches for i in batch]
            assert len(indices)==6851 and len(set(indices))==6851
    write(OUT/'export_verified.json',dict(verified=True,archive_sha256=sha(archive),kinds=kinds))

def cleanup():
    if (OUT/'cleanup_verified.json').exists():return
    assert read(OUT/'export_verified.json')['verified']
    result=cli('pod','get',POD)
    if result.returncode==0:
        pod=json.loads(result.stdout)
        assert pod['name']=='minimind-gsm8k-grpo-compare-20261025'
        cli('pod','delete',POD).check_returncode()
    elif 'not_found' not in result.stderr:
        raise RuntimeError('Cannot verify task pod state: '+result.stderr)
    result=cli('pod','get',POD)
    assert result.returncode!=0 and 'not_found' in result.stderr, result.stderr
    write(OUT/'cleanup_verified.json',dict(pod_id=POD,confirmed_absent=True,shared_volume_preserved=True,timestamp=time.time()))

def report(kinds):
    root=OUT/'results';records={};lines=['# GSM8K GRPO 보상 비교 결과','',
        '각 모델에서 두 보상 실험은 같은 Mixed-SFT 가중치에서 각각 시작했습니다. 규칙 보상 → Skywork 순서로 각 1epoch를 실행했습니다.',
        '학습 6,851문제 전체, 공식 테스트 1,319문제 전체. 테스트 정답은 학습·보상 모델 입력에 포함하지 않았습니다.','',
        '| 모델 | SFT 기준 정답 | 규칙 GRPO 정답 | Skywork GRPO 정답 |','|---|---:|---:|---:|']
    for kind in kinds:
        before=read(root/(kind+'-baseline/evaluation.json'))
        after={mode:read(root/(kind+'-'+mode+'/evaluation_after.json')) for mode in ['rule','skywork']}
        fmt=lambda s:f"{s['correct']}/1319 ({s['accuracy']:.2%})"
        lines.append(f"| {kind} | {fmt(before['summary'])} | {fmt(after['rule']['summary'])} | {fmt(after['skywork']['summary'])} |")
        records[kind]=dict(baseline=before['summary'],**{mode:a['summary'] for mode,a in after.items()})
    lines += ['', '## 실행 조건','',
        '- GPU: A40 48GB. 카탈로그/생성 후 조회 GPU 요금 $0.59/hour. 실제 청구액과 동일하다고 단정하지 않습니다.',
        '- 각 실험 215 optimizer updates, 질문 배치32 × 질문당4개 생성, 역전파 microbatch8.',
        '- LR 1e-6→1e-7 cosine, KL beta0.04, GRPO clip0.2, AdamW, seed42.',
        '- 규칙 보상은 정답1/나머지0. Skywork는 동결된 1.7B 모델의 원시 점수만 사용.',
        '- 입력 최대768토큰, 출력 최대256토큰. 질문을 자르지 않음. SFT의340토큰 문맥과 다르며 기존340토큰 평가와 직접 비교 불가.',
        '- 모든 평가 greedy. 같은 정답 채점 함수를 사용하며 풀이 과정의 타당성은 검증하지 않음.',
        '- 학습 중 저장·최종 재로드·가중치/순전파 일치 검증. 공개 패키지 평가 결과를 기록.',
        '- all-zero/mixed-group 비율, 형식 오류, 출력 한도, KL, gradient는 각 metrics.jsonl에 기록.',
        '', '## 고정 샘플 테스트','', '테스트 인덱스0·1·2를 고정 선택했습니다. 성공 사례만 고르지 않았습니다.']
    for kind in kinds:
        before=read(root/(kind+'-baseline/evaluation.json'))['rows']
        after={mode:read(root/(kind+'-'+mode+'/evaluation_after.json'))['rows'] for mode in ['rule','skywork']}
        for index in range(3):
            row=before[index]
            lines += ['',f"### {kind}, test index {row['index']}",'',row['question'],'','정답:','```text',row['answer'],'```']
            for name,rows in [('SFT',before),('규칙 GRPO',after['rule']),('Skywork GRPO',after['skywork'])]:
                sample=rows[index]
                lines += ['',f"**{name}** — 보상 {sample['reward']}, {sample['reason']}",'','```text',sample['response'],'```']
    if (root/'cross_reward_scores.json').exists():
        cross=read(root/'cross_reward_scores.json')['results']
        lines += ['', '## Skywork 보조 평가', '', '동일한 테스트 응답을 같은 Skywork 모델로 채점했습니다. 학습에 사용한 보상 모델이므로 독립적인 정답 지표는 아닙니다.', '',
                  '| 모델 | SFT 평균 점수 | 규칙 GRPO 평균 점수 | Skywork GRPO 평균 점수 |', '|---|---:|---:|---:|']
        for kind in kinds:
            values=[cross[kind+'-'+mode]['mean_skywork_reward'] for mode in ['baseline','rule','skywork']]
            lines.append(f'| {kind} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |')
    lines += ['', '## 초기 실행 오류와 복구', '', '초기 실행에서 생성 텐서의 역전파 제약과 상태 로그 중복 필드 오류를 수정했습니다. 실패 시도는 별도로 보존했고, 최종 Dense 규칙 실험은 원본 SFT에서 다시 시작해 정확히215업데이트를 완료했습니다. 수정한 동작에 회귀 테스트를 추가했습니다. Dense Skywork는 정책 확률 재계산의 수치 불일치 검사가 실패하여50번째 검증 체크포인트(모델·optimizer·RNG)에서 재개했습니다. 배치당1회 업데이트에 맞춰 이전 정책 log-probability를 현재 순전파에서 detach하는 동등한 계산으로 변경했고,50 이후 미저장 시도의 로그는 별도 rollback 기록으로 보존했습니다. 기존 호스트 GPU 재시작 불가로 같은 A40의 새 Pod에서 이어 실행했습니다.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'comparison.json',records)
    return records

def publish(kinds):
    api=HfApi();assert api.whoami()['name']=='caspar'
    comparison=read(OUT/'comparison.json')
    for kind in kinds:
        for mode in ['rule','skywork']:
            name=f'MiniMind-{ "Dense-64M" if kind=="dense" else "MoE-198M-A64M" }-GSM8K-GRPO-{mode.title()}-v1.0'
            repo='caspar/'+name
            marker=OUT/(kind+'-'+mode+'-repo-created.json')
            receipt=OUT/(kind+'-'+mode+'-published.json')
            if receipt.exists():continue
            run=OUT/'results'/(kind+'-'+mode)
            folder=run/'model';done=read(run/'completion.json');settings=read(run/'training_settings.json')
            for filename in ['completion.json','training_settings.json','evaluation_after.json']:
                shutil.copyfile(run/filename,folder/filename)
            shutil.copyfile(PROJECT/'trainer/gsm8k_reward.py',folder/'gsm8k_reward.py')
            shutil.copyfile(PROJECT/'scripts/runpod_gsm8k_grpo/run.py',folder/'training_recipe.py')
            shutil.copyfile(OUT/'dataset_manifest.json',folder/'dataset_manifest.json')
            cross_path=OUT/'results/cross_reward_scores.json'
            if cross_path.exists():
                cross=read(cross_path)
                write(folder/'reward_evaluation.json',dict(reward_model=cross['reward_model'],note=cross['note'],
                    baseline=cross['results'][kind+'-baseline'],final=cross['results'][kind+'-'+mode]))
            (folder/'requirements.txt').write_text('torch>=2.6,<3\ntransformers==4.57.6\nsafetensors>=0.4\n')
            b=done['baseline'];a=done['final']
            card=f'''---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
base_model: {done['base_repo']}
datasets:
- caspar/MiniMind-Mixed-SFT-Training-v1.0
tags:
- minimind
- grpo
- gsm8k
- experimental
---

# {name}

Full-parameter GRPO from `{done['base_repo']}` at `{done['base_revision']}`.
This is an experimental small language model, not a reliable arithmetic solver.

## Training

Reward variant: **{mode}**. The rule and Skywork variants start independently from
exactly the same SFT checkpoint, with matched data order and hyperparameters.
Rule reward: strict final `#### number` equality, correct1/otherwise0.
Skywork reward: frozen `Skywork/Skywork-Reward-V2-Qwen3-1.7B` raw classification
logit, without any rule reward, length bonus or repetition penalty. Skywork
revision `e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc`. Group-normalized advantages.

Dataset revision `542384d27d374d668413d8ade5b9973b7b063bee`: all6,851 retained
GSM8K train conversations. One full epoch,215updates, batch32questions,
four generations/question, microbatch8, AdamW LR1e-6→1e-7 cosine,
KL beta0.04, clipping0.2, weight decay0.01, gradient clip1, seed42.
FP32 parameters, BF16 autocast on A40. Sampling temperature1/top-p1/top-k0.
Prompts≤768tokens, output≤256tokens. No question truncation. Base SFT used
340-token windows; this run extends context and is not a matched340-token comparison.
No test gold is used in policy/reward prompts or optimization. Training gold is
used only for rule reward and diagnostic correctness, never Skywork input.

## Evaluation

All1,319 official GSM8K test questions, greedy, output≤256tokens, same strict
rule scorer before/after. Baseline **{b['correct']}/1319 ({b['accuracy']:.2%})**;
final **{a['correct']}/1319 ({a['accuracy']:.2%})**.
Final format errors: {a['counts'].get('format_error',0)}. Output budget exhausted:
{a['token_budget_exhausted']}. Full responses and gold in evaluation_after.json.
This verifies final numbers, not reasoning validity. No independent human
assessment is claimed. Training reward alone does not establish improved math.
The final FP32 safetensors package was reloaded and exact weight/logit equality
was checked, then the reloaded model was evaluated. No best-checkpoint selection.

## Usage

Review model_minimind.py before enabling custom code.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
repo = "{repo}"
tok = AutoTokenizer.from_pretrained(repo, revision="v1.0")
model = AutoModelForCausalLM.from_pretrained(repo, revision="v1.0", trust_remote_code=True).eval()
question = "What is 2 + 2?\\nSolve step by step. End your answer with #### followed by the final number."
prompt = tok.apply_chat_template([{{"role":"user","content":question}}], tokenize=False, add_generation_prompt=True, open_thinking=False).replace('<think>\\n\\n</think>\\n\\n', '')
x = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
with torch.inference_mode():
    y = model.generate(inputs=x, max_new_tokens=256, do_sample=False, temperature=1., top_p=1., top_k=0, logits_to_keep=1)
print(tok.decode(y[0,x.shape[1]:], skip_special_tokens=True))
```

## Attribution

MiniMind code: https://github.com/jingyaogong/minimind (Apache-2.0).
GSM8K is MIT licensed; upstream mixed-SFT dataset component terms remain
applicable. Skywork reward model: https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-1.7B
(Apache-2.0); its weights are not redistributed in this repository.
'''
            (folder/'README.md').write_text(card)
            write(folder/'release.json',dict(version='v1.0',files={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='release.json'}))
            if not marker.exists():
                api.create_repo(repo,private=False,exist_ok=False)
                write(marker,dict(repo=repo,created_for_this_task=True))
            commit=api.upload_folder(repo_id=repo,folder_path=folder,commit_message='Release verified full-epoch GSM8K GRPO comparison')
            api.create_tag(repo,tag='v1.0',revision=commit.oid,exist_ok=True)
            public=HfApi(token=False)
            info=public.model_info(repo,revision=commit.oid,files_metadata=True)
            assert not info.private and info.sha==commit.oid
            files={p.rfilename:p for p in info.siblings}
            weights=files['model.safetensors'];assert weights.lfs.sha256==sha(folder/'model.safetensors')
            for filename in ['config.json','README.md','evaluation_after.json','release.json']:
                downloaded=hf_hub_download(repo,filename,revision=commit.oid,token=False,cache_dir=str(OUT/'verify-cache'))
                assert sha(downloaded)==sha(folder/filename)
            assert public.model_info(repo,revision='v1.0').sha==commit.oid
            write(receipt,dict(repo=repo,url='https://huggingface.co/'+repo,commit=commit.oid,
                public=True,files_verified=True,tag='v1.0',accuracy=a['accuracy']))
            print('PUBLISHED',repo,flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    while not (OUT/'export_verified.json').exists():
        try:
            result=ssh('cat '+ROOT+'/status.json',capture_output=True,text=True,timeout=45)
            if result.returncode:
                raise RuntimeError(result.stderr[-1000:])
            state=json.loads(result.stdout);write(OUT/'live_status.json',state)
            print(json.dumps(state),flush=True)
            stale=time.time()-state.get('timestamp',time.time())>1800
            if state['phase']=='failed' or stale:
                log=ssh('tail -n 100 '+ROOT+'/bootstrap*.log',capture_output=True,text=True,timeout=45)
                (OUT/'failure.log').write_text(log.stdout+log.stderr)
                # This task's output/checkpoints are durable; stop only this task Pod.
                cli('pod','stop',POD).check_returncode()
                fresh=cli('pod','get',POD);fresh.check_returncode();p=json.loads(fresh.stdout)
                write(OUT/'stopped_on_failure.json',dict(pod_id=POD,status=p.get('desiredStatus'),timestamp=time.time()))
                raise SystemExit('Training failed or heartbeat stale; task Pod stopped; durable outputs preserved')
            if state['phase']=='all_complete':
                requested = read(OUT/'requested_kinds.json') if (OUT/'requested_kinds.json').exists() else ['dense']
                if state['kinds'] != requested:
                    assert state['kinds'] == ['dense'] and requested == ['dense','moe']
                    command = ('setsid bash /tmp/minimind-gsm8k-grpo-source/scripts/runpod_gsm8k_grpo/bootstrap.sh dense moe > '
                               +ROOT+'/bootstrap-moe-continuation.log 2>&1 < /dev/null & echo MOE_CONTINUATION_LAUNCHED')
                    launched=ssh(command,capture_output=True,text=True,timeout=45)
                    launched.check_returncode()
                    write(OUT/'moe_continuation_launched.json',dict(timestamp=time.time(),kinds=requested))
                    time.sleep(45)
                    continue
                export(state['kinds']);break
        except (OSError,RuntimeError,ValueError,subprocess.SubprocessError) as e:
            write(OUT/'supervisor_error.json',dict(type=type(e).__name__,detail=str(e),timestamp=time.time()))
            print('SUPERVISOR_RETRY',type(e).__name__,str(e),flush=True)
        time.sleep(45)
    kinds=read(OUT/'export_verified.json')['kinds']
    cleanup()
    report(kinds)
    publish(kinds)
    write(OUT/'completion.json',dict(complete=True,kinds=kinds,all_task_pods_terminated=True,published=True,timestamp=time.time()))
    print('ALL_COMPLETE',flush=True)

if __name__=='__main__':
    main()
