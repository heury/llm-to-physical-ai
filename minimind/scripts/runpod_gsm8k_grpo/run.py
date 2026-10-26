"""Sequential, matched-base GRPO comparison; full GSM8K training split."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from trainer.gsm8k_reward import score_gsm8k_response

ROOT = Path('/shared-volume/minimind-gsm8k-grpo-20261025')
BASES = {
    'dense': ('caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0', '7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d'),
    'moe': ('caspar/MiniMind-MoE-198M-A64M-Mixed-SFT-v1.0', 'a742772a0ad4db7480064bd2abccc174062a3de8'),
}
REWARD = ('Skywork/Skywork-Reward-V2-Qwen3-1.7B', 'e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc')
SETTINGS = dict(epochs=1, batch_size=32, generations=4, micro_batch_size=8, learning_rate=1e-6,
                beta=.04, epsilon=.2, max_prompt_tokens=768, max_new_tokens=256, seed=42,
                temperature=1., top_p=1., top_k=0, grad_clip=1., weight_decay=.01,
                dtype='bfloat16 autocast; fp32 parameters and optimizer', loss='GRPO',
                grouping='prompt-major; length-sorted batches shuffled with seed42',
                old_logps='single on-policy update; detach current forward before optimizer step',
                reward_rule='exact shared score_gsm8k_response; correct=1 otherwise=0',
                reward_skywork='raw frozen classification logit; no rule bonus, no clipping, no length/repetition shaping',
                evaluation='all1319 official test questions; greedy; max256 new tokens; full question retained',
                comparison='each run starts independently from the same frozen mixed-SFT base',
                moe_auxiliary='native router load-balancing loss during backward forward; coefficient from base config')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def status(phase, **kwargs):
    value = dict(phase=phase, **kwargs)
    value['timestamp'] = time.time()
    write(ROOT/'status.json', value)
    print(json.dumps(value), flush=True)


def seed():
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def autocast():
    return torch.autocast('cuda', dtype=torch.bfloat16)


def load_base(path):
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
            local_files_only=True, torch_dtype=torch.float32).cuda().eval()
    assert model.config.dropout == 0
    return model


def encode_rows(tok, split):
    with (ROOT/(split+'.jsonl')).open() as f:
        rows = [json.loads(line) for line in f]
    for row in rows:
        row['prompt'] = tok.apply_chat_template([dict(role='user', content=row['question'])],
            tokenize=False, add_generation_prompt=True, open_thinking=False).replace('<think>\n\n</think>\n\n', '')
        row['ids'] = tok(row['prompt'], add_special_tokens=False).input_ids
        assert len(row['ids']) <= SETTINGS['max_prompt_tokens'], (row['index'], len(row['ids']))
        score_gsm8k_response('#### 0', row['answer'])  # fail early on malformed gold
    return rows


def batch_inputs(rows, tok, generations=1):
    length = max(len(r['ids']) for r in rows)
    ids = torch.tensor([[tok.pad_token_id]*(length-len(r['ids']))+r['ids'] for r in rows], device='cuda')
    mask = torch.tensor([[0]*(length-len(r['ids']))+[1]*len(r['ids']) for r in rows], device='cuda')
    return ids.repeat_interleave(generations, 0), mask.repeat_interleave(generations, 0)


def mask_completion(ids, eos_id):
    eos = ids.eq(eos_id)
    return (eos.cumsum(1)-eos.long()).eq(0)


def logps(model, ids, mask, count):
    result = model(ids, attention_mask=mask, logits_to_keep=count+1)
    logits = result.logits[:, :-1].float()
    result_logps = logits.log_softmax(-1).gather(-1, ids[:, -count:].unsqueeze(-1)).squeeze(-1)
    aux = result.aux_loss if model.config.use_moe else result_logps.new_zeros(())
    return result_logps, aux


class Skywork:
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path,
            local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa', num_labels=1).cuda().eval().requires_grad_(False)

    @torch.no_grad()
    def score(self, questions, responses):
        texts = []
        for question, response in zip(questions, responses):
            text = self.tokenizer.apply_chat_template([dict(role='user', content=question),
                dict(role='assistant', content=response)], tokenize=False, add_generation_prompt=False)
            bos = self.tokenizer.bos_token
            if bos and text.startswith(bos):
                text = text[len(bos):]
            texts.append(text)
        scores = []
        for start in range(0, len(texts), 8):
            inputs = self.tokenizer(texts[start:start+8], padding=True, truncation=False, return_tensors='pt').to('cuda')
            assert inputs['input_ids'].shape[1] <= 16384, 'Never silently truncate reward inputs'
            scores.append(self.model(**inputs).logits[:, 0].float())
        result = torch.cat(scores)
        assert torch.isfinite(result).all()
        return result


@torch.no_grad()
def evaluate(model, tok, rows, out, label):
    path = out/(label+'.json')
    if path.exists():
        saved = json.loads(path.read_text())
        assert len(saved['rows']) == len(rows)
        return saved['summary']
    groups = defaultdict(list)
    for i, row in enumerate(sorted(rows, key=lambda r:len(r['ids']))):
        groups[i // 32].append(row)
    results = []
    started = time.time()
    for length, group in sorted(groups.items()):
        for start in range(0, len(group), 32):
            batch = group[start:start+32]
            x, mask = batch_inputs(batch, tok)
            length = x.size(1)
            with autocast():
                y = model.generate(inputs=x, attention_mask=mask, max_new_tokens=256,
                    do_sample=False, temperature=1., top_p=1., top_k=0, repetition_penalty=1.,
                    eos_token_id=tok.eos_token_id, logits_to_keep=1)
            completion = y[:, x.size(1):]
            for row, tokens in zip(batch, completion.tolist()):
                ended = tok.eos_token_id in tokens
                if ended:
                    tokens = tokens[:tokens.index(tok.eos_token_id)+1]
                response = tok.decode(tokens, skip_special_tokens=True)
                reward, reason = score_gsm8k_response(response, row['answer'])
                results.append(dict(index=row['index'], question=row['question'], answer=row['answer'],
                    response=response, reward=reward, reason=reason, prompt_tokens=len(row['ids']), padded_prompt_tokens=length,
                    generated_tokens=len(tokens), eos_reached=ended, token_budget_exhausted=not ended and len(tokens)>=256))
            if len(results) % 32 < len(batch):
                status('evaluation', label=label, run=out.name, completed=len(results), total=len(rows), elapsed=time.time()-started)
    counts = Counter(r['reason'] for r in results)
    summary = dict(rows=len(rows), correct=counts['correct'], accuracy=counts['correct']/len(rows),
        counts=dict(counts), token_budget_exhausted=sum(r['token_budget_exhausted'] for r in results), seconds=time.time()-started)
    write(path, dict(summary=summary, rows=sorted(results, key=lambda r:r['index'])))
    return summary


def save_checkpoint(model, optimizer, step, out):
    # Publish a fresh file, then hash it; keep optimizer state on durable storage.
    local = Path('/tmp')/('gsm8k-grpo-'+out.name+'.pth')
    state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in state.values())
    torch.save(dict(model=state, optimizer=optimizer.state_dict(), step=step, settings=SETTINGS,
                    rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), local)
    target = out/f'checkpoint_{step:05d}.pth'
    shutil.copyfile(local, target)
    digest = sha(local)
    assert sha(target) == digest
    write(out/'checkpoint.json', dict(step=step, path=str(target), sha256=digest))
    previous = sorted(out.glob('checkpoint_*.pth'))
    for p in previous[:-2]:
        p.unlink()
    local.unlink()
    return target


def train(model, ref, tok, rows, scorer, mode, out):
    seed()
    ordered = sorted(range(len(rows)), key=lambda i:len(rows[i]['ids']))
    batches = [ordered[start:start+SETTINGS['batch_size']] for start in range(0, len(rows), SETTINGS['batch_size'])]
    random.Random(42).shuffle(batches)
    write(out/'batch_indices.json', [[rows[i]['index'] for i in batch] for batch in batches])
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS['learning_rate'], weight_decay=.01)
    first = 0
    if (out/'checkpoint.json').exists():
        receipt = json.loads((out/'checkpoint.json').read_text())
        assert sha(receipt['path']) == receipt['sha256']
        saved = torch.load(receipt['path'], map_location='cpu', weights_only=True)
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        first = saved['step']
        del saved
    model.eval()  # dropout=0; deterministic likelihood and generation paths
    previous_elapsed = 0.
    if first and (out/'metrics.jsonl').exists():
        records = [json.loads(line) for line in (out/'metrics.jsonl').open()]
        retained = [r for r in records if r['step'] <= first]
        discarded = [r for r in records if r['step'] > first]
        if discarded:
            write(out/f'rollback_after_{first:05d}.json', dict(discarded_metrics=discarded,
                reason='Restore model, optimizer and RNG from last hash-verified checkpoint'))
        (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in retained))
        previous_elapsed = retained[-1]['elapsed'] if retained else 0.
    started = time.time() - previous_elapsed
    for step, indices in enumerate(batches, 1):
        if step <= first:
            continue
        begin = time.time()
        batch = [rows[i] for i in indices]
        x, prompt_mask = batch_inputs(batch, tok, SETTINGS['generations'])
        with torch.no_grad(), autocast():
            outputs = model.generate(inputs=x, attention_mask=prompt_mask, max_new_tokens=SETTINGS['max_new_tokens'],
                do_sample=True, temperature=1., top_k=0, top_p=1., repetition_penalty=1.,
                eos_token_id=tok.eos_token_id, logits_to_keep=1)
        outputs = outputs.clone()  # generate() returns inference-mode tensors
        completion = outputs[:, x.size(1):]
        mask = mask_completion(completion, tok.eos_token_id)
        full_mask = torch.cat([prompt_mask, mask.long()], 1)
        texts = tok.batch_decode(completion, skip_special_tokens=True)
        questions = [r['question'] for r in batch for _ in range(SETTINGS['generations'])]
        gold = [r['answer'] for r in batch for _ in range(SETTINGS['generations'])]
        exact = [score_gsm8k_response(text, answer) for text, answer in zip(texts, gold)]
        rewards = (torch.tensor([v for v, reason in exact], device='cuda', dtype=torch.float32)
                   if mode == 'rule' else scorer.score(questions, texts))
        groups = rewards.view(-1, SETTINGS['generations'])
        std = groups.std(1, unbiased=False)
        advantages = ((groups-groups.mean(1, keepdim=True))/(std[:,None]+1e-4)).flatten()
        n = len(texts)
        optimizer.zero_grad(set_to_none=True)
        total_loss = total_kl = ratio_error = 0.
        lr = SETTINGS['learning_rate']*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/len(batches))))
        for group in optimizer.param_groups:
            group['lr'] = lr
        # Compute all likelihoods with identical microbatch shapes, masks and autocast.
        for start in range(0, n, SETTINGS['micro_batch_size']):
            end = min(n, start+SETTINGS['micro_batch_size'])
            ids, fm, cm = outputs[start:end], full_mask[start:end], mask[start:end]
            with torch.no_grad(), autocast():
                reference, _ = logps(ref, ids, fm, completion.size(1))
            model.train()  # preserve native MoE router load-balancing loss; dropout is zero
            with autocast():
                current, aux = logps(model, ids, fm, completion.size(1))
            model.eval()
            # One optimizer update per rollout, no policy reuse: theta_old == theta.
            # Detach the same forward to avoid precision/backend drift from a redundant pass.
            old = current.detach()
            ratio = (current-old).exp()
            ratio_error = max(ratio_error, (ratio.detach()-1).abs().max().item())
            delta = reference-current
            kl = delta.exp()-delta-1
            a = advantages[start:end, None]
            surrogate = torch.minimum(ratio*a, ratio.clamp(.8, 1.2)*a)
            loss = (((-surrogate+SETTINGS['beta']*kl)*cm).sum(1)/cm.sum(1).clamp_min(1)).mean()
            loss = (loss+aux)*(end-start)/n
            assert torch.isfinite(loss), 'Nonfinite GRPO loss'
            loss.backward()
            total_loss += loss.detach().item()
            total_kl += ((kl.detach()*cm).sum(1)/cm.sum(1)).sum().item()/n
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS['grad_clip'])
        assert math.isfinite(grad.item()), 'Nonfinite gradient'
        assert ratio_error < 1e-3, ('Likelihood mismatch', ratio_error)
        if step == 1:
            assert total_kl < 1e-4, ('Initial reference KL must be zero', total_kl)
        optimizer.step()
        assert all(p.grad is None for p in ref.parameters())
        counts = Counter(reason for _, reason in exact)
        record = dict(step=step, steps=len(batches), prompts=len(batch), reward_mean=rewards.mean().item(),
            rule_accuracy=counts['correct']/n, format_error_rate=counts['format_error']/n,
            mixed_group_rate=(std>1e-6).float().mean().item(), all_zero_group_rate=(groups==0).all(1).float().mean().item(),
            policy_loss=total_loss, reference_kl=total_kl, ratio_error=ratio_error, grad_norm=grad.item(), learning_rate=lr,
            response_length=mask.sum(1).float().mean().item(), token_budget_exhausted_rate=(~completion.eq(tok.eos_token_id).any(1)).float().mean().item(),
            seconds=time.time()-begin, elapsed=time.time()-started, cuda_peak_bytes=torch.cuda.max_memory_allocated(), timestamp=time.time())
        with (out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        if step == 1 or step % 25 == 0 or step == len(batches):
            write(out/f'train_samples_{step:05d}.json', [dict(question=q, answer=a, response=t, reward=float(r), rule_reward=e[0], reason=e[1])
                for q,a,t,r,e in zip(questions[:8],gold[:8],texts[:8],rewards[:8].tolist(),exact[:8])])
        status('training', run=out.name, **record)
        if step == 1 or step % 50 == 0 or step == len(batches):
            save_checkpoint(model, optimizer, step, out)
        del outputs, completion, mask, full_mask, rewards, groups, advantages, current, old, reference, loss, ratio, kl
    final = json.loads((out/'checkpoint.json').read_text())
    assert final['step'] == len(batches)
    saved = torch.load(final['path'], map_location='cpu', weights_only=True)
    assert all(torch.equal(v.detach().cpu(), saved['model'][k]) for k,v in model.state_dict().items())
    assert all(torch.isfinite(v).all() for state in saved['optimizer']['state'].values() for v in state.values() if torch.is_tensor(v))
    return dict(optimizer_updates=len(batches), training_seconds=time.time()-started,
                checkpoint_verified=True, checkpoint_sha256=final['sha256'])


def package(model, tok, out, base_path):
    folder = out/'model'
    model.config.finetuning_task = 'gsm8k-grpo-'+out.name
    # FP32 release preserves exact final weights; tiny policy keeps artifacts small.
    model.save_pretrained(folder, safe_serialization=True)
    tok.save_pretrained(folder)
    restored = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True,
                 local_files_only=True, torch_dtype=torch.float32).cuda().eval()
    assert all(torch.equal(v, restored.state_dict()[k]) for k,v in model.state_dict().items())
    x = torch.tensor([[1,20,21,2]], device='cuda')
    with torch.no_grad(), autocast():
        torch.testing.assert_close(model(x).logits, restored(x).logits, atol=0, rtol=0)
    shutil.copyfile(base_path/'LICENSE', folder/'LICENSE')
    del restored
    return folder



def cross_evaluate_rewards(kinds):
    """Post-training scoring only; never feed test results back into optimization."""
    target = ROOT/'cross_reward_scores.json'
    if target.exists():
        return
    status('cross_reward_evaluation', kinds=kinds)
    reward_path = snapshot_download(REWARD[0], revision=REWARD[1], token=False, cache_dir=str(ROOT/'hf-cache'),
        allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])
    scorer = Skywork(reward_path)
    results = {}
    for kind in kinds:
        for mode in ['baseline','rule','skywork']:
            name = kind+'-'+mode
            filename = 'evaluation.json' if mode == 'baseline' else 'evaluation_after.json'
            rows = json.loads((ROOT/name/filename).read_text())['rows']
            assert len(rows) == 1319
            values = []
            for start in range(0, len(rows), 64):
                batch = rows[start:start+64]
                values.extend(scorer.score([r['question'] for r in batch], [r['response'] for r in batch]).tolist())
                status('cross_reward_evaluation', run=name, completed=len(values), total=len(rows))
            results[name] = dict(mean_skywork_reward=sum(values)/len(values), rows=len(values),
                correct=sum(r['reward'] for r in rows),
                scores=[dict(index=r['index'],skywork_reward=v,rule_reward=r['reward']) for r,v in zip(rows,values)])
    write(target, dict(reward_model=REWARD, note='Learned reward is not an independent correctness metric', results=results))
    del scorer
    gc.collect();torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kinds', nargs='+', default=['dense'], choices=['dense','moe'])
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = True
    write(ROOT/'settings.json', dict(SETTINGS, kinds=args.kinds, bases=BASES, reward_model=REWARD))
    for split, expected in [('train',6851), ('test',1319)]:
        manifest = json.loads((ROOT/'dataset_manifest.json').read_text())
        assert sha(ROOT/(split+'.jsonl')) == manifest['splits'][split]['sha256']
        assert manifest['splits'][split]['rows'] == expected
    for kind in args.kinds:
        status('download_base', kind=kind)
        repo, revision = BASES[kind]
        base = Path(snapshot_download(repo, revision=revision, token=False, cache_dir=str(ROOT/'hf-cache')))
        assert sha(base/'model_minimind.py') == sha(PROJECT/'model/model_minimind.py'), 'Review changed base code before loading'
        tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
        assert len(tok) == 6400 and tok.pad_token_id == 0
        train_rows, test_rows = encode_rows(tok,'train'), encode_rows(tok,'test')
        assert not {r['question'] for r in train_rows} & {r['question'] for r in test_rows}
        write(ROOT/(kind+'-token_lengths.json'), {split:dict(rows=len(rows), min=min(len(r['ids']) for r in rows), max=max(len(r['ids']) for r in rows))
                                               for split,rows in [('train',train_rows),('test',test_rows)]})
        baseline_dir = ROOT/(kind+'-baseline')
        model = load_base(base)
        baseline = evaluate(model,tok,test_rows,baseline_dir,'evaluation')
        del model
        gc.collect();torch.cuda.empty_cache()
        for mode in ['rule','skywork']:
            out = ROOT/(kind+'-'+mode)
            out.mkdir(exist_ok=True)
            if (out/'completion.json').exists():
                continue
            seed()
            model = load_base(base)
            ref = load_base(base).requires_grad_(False)
            assert all(torch.equal(v,ref.state_dict()[k]) for k,v in model.state_dict().items())
            scorer = None
            if mode == 'skywork':
                status('download_reward_model', run=out.name)
                reward_path = snapshot_download(REWARD[0], revision=REWARD[1], token=False, cache_dir=str(ROOT/'hf-cache'),
                    allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])
                scorer = Skywork(reward_path)
                sanity_questions = ['What is 2 + 2?']*2
                sanity_responses = ['2 + 2 = 4.\n#### 4', '2 + 2 = 99.\n#### 99']
                sanity = scorer.score(sanity_questions, sanity_responses).tolist()
                write(out/'scorer_sanity.json', dict(questions=sanity_questions,responses=sanity_responses,scores=sanity))
                assert sanity[0] != sanity[1], 'Reward model returned identical sanity scores'
            write(out/'training_settings.json', dict(SETTINGS, mode=mode, kind=kind, base_repo=repo, base_revision=revision,
                base_weights_sha256=sha(base/'model.safetensors'), reward_model=REWARD if mode=='skywork' else None))
            report = train(model,ref,tok,train_rows,scorer,mode,out)
            del ref, scorer
            gc.collect();torch.cuda.empty_cache()
            folder = package(model,tok,out,base)
            del model
            gc.collect();torch.cuda.empty_cache()
            model = load_base(folder)
            final = evaluate(model,tok,test_rows,out,'evaluation_after')
            write(out/'completion.json', dict(complete=True, base_repo=repo, base_revision=revision, kind=kind, mode=mode,
                epochs=1, train_rows=6851, test_rows=1319, baseline=baseline, final=final,
                package_verified=True, model_files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}, **report))
            status('run_complete', run=out.name, before=baseline['accuracy'], after=final['accuracy'])
            del model
            gc.collect();torch.cuda.empty_cache()
    if len(args.kinds) > 1:
        cross_evaluate_rewards(args.kinds)
    status('all_complete', kinds=args.kinds)
    write(ROOT/'all_complete.json', dict(complete=True,kinds=args.kinds,timestamp=time.time()))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        status('failed', traceback=traceback.format_exc())
        raise
