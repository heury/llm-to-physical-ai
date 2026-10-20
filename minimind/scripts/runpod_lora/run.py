"""Reproducible ChatDoctor LoRA job; native MiniMind model, actual W&B logging."""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora

ROOT = Path('/shared-volume/minimind-lora-chatdoctor-20261020')
RUNTIME = Path('/tmp/minimind-lora-chatdoctor-20261020')
BASE_ROOT = Path('/shared-volume/minimind-sft-ultrachat-200k-20261020')
BASE = BASE_ROOT / 'out/full_sft_768.pth'
REVISION = '505443eac4e99ccedeffbb6f640061223d1d4bb3'
SETTINGS = dict(dataset='lavita/ChatDoctor-HealthCareMagic-100k', revision=REVISION,
                epochs=1, batch_size=16, accumulation_steps=2, learning_rate=1e-4,
                max_seq_len=1024, rank=16, seed=42, save_interval=250,
                hidden_size=768, num_hidden_layers=8, memory_fraction=0.17)


def write(name, value):
    path = ROOT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def phase(name):
    write('phase.json', dict(phase=name, updated=time.time()))


def atomic_save(value, path):
    path = Path(path)
    torch.save(value, str(path) + '.tmp')
    os.replace(str(path) + '.tmp', path)


def prepare():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ['data', 'tokenizer', 'out', 'checkpoints']:
        (ROOT / name).mkdir(exist_ok=True)
    phase('preparing')
    assert sha(BASE) == json.loads((BASE_ROOT / 'evaluation_after.json').read_text())['checkpoint_sha256']
    for name in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']:
        path = BASE_ROOT / 'tokenizer' / name
        if path.exists(): shutil.copyfile(path, ROOT / 'tokenizer' / name)
    shutil.copyfile(BASE_ROOT / 'config.json', ROOT / 'config.json')
    tok = AutoTokenizer.from_pretrained(ROOT / 'tokenizer', local_files_only=True)
    file = hf_hub_download(SETTINGS['dataset'], 'data/train-00000-of-00001-5e7cb295b9cff0bf.parquet',
                          repo_type='dataset', revision=REVISION, token=False,
                          cache_dir=str(ROOT / 'hf-cache'))
    rows = pq.read_table(file).to_pylist()
    # Disk-backed fixed-width arrays avoid duplicating the entire corpus in RAM.
    arrays = {key: np.lib.format.open_memmap(RUNTIME / (key + '.npy'), mode='w+',
              dtype=np.int16, shape=(len(rows), 1024)) for key in ['ids', 'labels']}
    counts = dict(source=len(rows), empty=0, duplicate=0, short_response=0, retained=0)
    seen = set(); train = []; validation = []; sample = None
    with (ROOT / 'data/conversations.jsonl').open('w') as out:
        for index, row in enumerate(rows):
            question = str(row.get('input') or '').strip()
            answer = str(row.get('output') or '').strip()
            instruction = str(row.get('instruction') or '').strip() or 'You are a helpful assistant.'
            if not question or not answer:
                counts['empty'] += 1; continue
            identity = hashlib.sha256(question.encode()).hexdigest()
            # Keep identical questions in exactly one split and avoid duplicate supervision.
            if identity in seen:
                counts['duplicate'] += 1; continue
            seen.add(identity)
            messages = [dict(role='system', content=instruction), dict(role='user', content=question), dict(role='assistant', content=answer)]
            prefix = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, open_thinking=False)
            full = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, open_thinking=False)
            assert full[:len(prefix)] == prefix, 'Assistant prefix is not aligned'
            ids = full[:1024]
            labels = [-100] * min(len(prefix), len(ids)) + ids[len(prefix):]
            if sum(v != -100 for v in labels[1:]) < 32:
                counts['short_response'] += 1; continue
            length = len(ids)
            ids += [tok.pad_token_id] * (1024 - length)
            labels += [-100] * (1024 - length)
            assert len(ids) == len(labels) == 1024
            assert all(v == -100 for v in labels[:len(prefix)])
            assert all(v == -100 for v in labels[length:])
            n = counts['retained']; arrays['ids'][n] = ids; arrays['labels'][n] = labels
            split = 'validation' if int(identity[:8], 16) % 100 == 0 else 'train'
            (validation if split == 'validation' else train).append(n)
            out.write(json.dumps(dict(split=split, prompt_sha256=identity, conversations=messages), ensure_ascii=False) + '\n')
            if sample is None: sample = dict(prompt=question, decoded_target=tok.decode([x for x in labels if x != -100]), assistant_tokens=sum(x != -100 for x in labels))
            counts['retained'] += 1
            if index % 10000 == 0: print('prepare', index, counts, flush=True)
    for array in arrays.values(): array.flush()
    assert len(train) > 90000 and len(validation) >= 256
    np.save(RUNTIME / 'train_indices.npy', np.array(train, dtype=np.int64))
    np.save(RUNTIME / 'validation_indices.npy', np.array(validation[:256], dtype=np.int64))
    for name in ['ids.npy', 'labels.npy', 'train_indices.npy', 'validation_indices.npy']:
        shutil.copyfile(RUNTIME / name, ROOT / 'data' / name)
    write('dataset_manifest.json', dict(**counts, train=len(train), validation_pool=len(validation), validation=256,
          validation_rule='SHA256(input) first32bits mod100==0; first256 retained; no shared question hashes',
          dataset=SETTINGS['dataset'], revision=REVISION, parquet_sha256=sha(file),
          base_sha256=sha(BASE), mask_sample=sample, steps_per_epoch=math.ceil(len(train)/16)))
    write('training_settings.json', SETTINGS)
    phase('prepared')


class Tokens(Dataset):
    def __init__(self, split):
        self.ids = np.load(RUNTIME / 'ids.npy', mmap_mode='r')
        self.labels = np.load(RUNTIME / 'labels.npy', mmap_mode='r')
        self.indices = np.load(RUNTIME / (split + '_indices.npy'))

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return torch.tensor(self.ids[j], dtype=torch.long), torch.tensor(self.labels[j], dtype=torch.long)


def load_model():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(SETTINGS['memory_fraction'])
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
    weights = torch.load(BASE, map_location='cpu', weights_only=True)
    assert all(torch.isfinite(v).all() for v in weights.values())
    model.load_state_dict(weights, strict=True)
    model.cuda(); apply_lora(model, rank=16)
    for name, parameter in model.named_parameters(): parameter.requires_grad_('.lora.' in name)
    assert all('.lora.' in name for name, p in model.named_parameters() if p.requires_grad)
    return model


def adapter(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if '.lora.' in n}


@torch.inference_mode()
def evaluate(model):
    model.eval(); total = 0.; count = 0
    for x, y in DataLoader(Tokens('validation'), batch_size=8, num_workers=0):
        x = x.cuda(); y = y.cuda(); n = int((y[:, 1:] != -100).sum())
        with torch.autocast('cuda', dtype=torch.bfloat16): result = model(x, labels=y)
        assert torch.isfinite(result.loss)
        total += float(result.loss) * n; count += n
    model.train()
    return dict(loss=total/count, perplexity=math.exp(total/count), tokens=count, documents=256)


def save(model, optimizer, step, run_id, final=False):
    weights = adapter(model)
    assert weights and all(torch.isfinite(v).all() for v in weights.values())
    atomic_save(weights, ROOT / 'out/lora_medical_768.pth')
    checkpoint = dict(adapter=weights, optimizer=optimizer.state_dict(), step=step, wandb_id=run_id,
                      settings=SETTINGS, final=final)
    assert all(torch.isfinite(v).all() for state in checkpoint['optimizer']['state'].values() for v in state.values() if torch.is_tensor(v))
    atomic_save(checkpoint, ROOT / 'checkpoints/lora_medical_768_resume.pth')
    write('checkpoint.json', dict(step=step, final=final, wandb_id=run_id,
          adapter_sha256=sha(ROOT / 'out/lora_medical_768.pth')))


def smoke():
    phase('smoke'); model = load_model(); model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    initial = adapter(model)
    x, y = next(iter(DataLoader(Tokens('train'), batch_size=16)))
    torch.cuda.reset_peak_memory_stats(); start = time.time()
    with torch.autocast('cuda', dtype=torch.bfloat16): loss = model(x.cuda(), labels=y.cuda()).loss
    assert torch.isfinite(loss); loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(parameters, 1.)
    assert torch.isfinite(grad) and grad > 0
    assert all(p.grad is None for n, p in model.named_parameters() if '.lora.' not in n)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4); optimizer.step()
    after = adapter(model)
    changed = sum(not torch.equal(initial[n], v) for n, v in after.items()); assert changed > 0
    base = torch.load(BASE, map_location='cpu', weights_only=True)
    assert all(torch.equal(v.cpu(), base[n].float()) for n, v in model.state_dict().items() if '.lora.' not in n)
    write('smoke.json', dict(loss=float(loss), grad_norm=float(grad), batch_size=16,
          peak_cuda_bytes=torch.cuda.max_memory_allocated(), seconds=time.time()-start,
          adapter_tensors_changed=changed, base_unchanged=True, disposable_model=True,
          trainable_parameters=sum(p.numel() for p in parameters)))
    phase('smoke_passed')


def train():
    import wandb
    with (ROOT/'requirements-resolved.txt').open('w') as file:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=file, check=True)
    phase('baseline_evaluation'); model = load_model()
    before = evaluate(model); write('evaluation_before.json', before)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    loader = DataLoader(Tokens('train'), batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(42), num_workers=0)
    total = len(loader)
    with wandb.init(project='MiniMind-ChatDoctor-LoRA', name='ChatDoctor-768-r16-1epoch', config=SETTINGS) as run:
        write('wandb_run.json', dict(id=run.id, url=run.url, project=run.project))
        run.log({'validation/loss': before['loss'], 'validation/perplexity': before['perplexity']}, step=0)
        phase('training'); start = time.time(); interval_start = start; optimizer.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(loader, 1):
            lr = 1e-4 * (.1 + .45 * (1 + math.cos(math.pi * step / total)))
            for group in optimizer.param_groups: group['lr'] = lr
            # Normalize a final incomplete accumulation window by its actual size.
            window_size = min(2, total - ((step-1)//2)*2)
            with torch.autocast('cuda', dtype=torch.bfloat16): loss = model(x.cuda(), labels=y.cuda()).loss
            assert torch.isfinite(loss); (loss/window_size).backward()
            if step % 2 == 0 or step == total:
                grad = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                assert torch.isfinite(grad); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 10 == 0 or step == total:
                metrics = dict(loss=float(loss), learning_rate=lr, step=step, total_steps=total,
                    progress_percent=step/total*100, eta_minutes=(time.time()-start)/step*(total-step)/60,
                    elapsed_seconds=time.time()-start, peak_cuda_bytes=torch.cuda.max_memory_allocated())
                if step > 1: metrics['grad_norm'] = float(grad)
                run.log(metrics, step=step); write('progress.json', metrics)
                print(json.dumps(metrics), flush=True)
            if step == 2 or step % 250 == 0 or step == total:
                save(model, optimizer, step, run.id, final=step==total)
            if step % 1000 == 0:
                validation = evaluate(model)
                run.log({'validation/loss': validation['loss'], 'validation/perplexity': validation['perplexity']}, step=step)
        phase('final_evaluation'); after = evaluate(model)
        run.log({'validation/loss': after['loss'], 'validation/perplexity': after['perplexity']}, step=total+1)
        base = torch.load(BASE, map_location='cpu', weights_only=True)
        assert all(torch.equal(v.cpu(), base[n].float()) for n, v in model.state_dict().items() if '.lora.' not in n)
        weights = adapter(model)
        reloaded = torch.load(ROOT/'out/lora_medical_768.pth', map_location='cpu', weights_only=True)
        assert weights.keys() == reloaded.keys() and all(torch.equal(v, reloaded[n]) for n,v in weights.items())
        # Merge in FP32, then serialize standard FP16 standalone model weights.
        merged = {n: v.detach().cpu().clone() for n,v in model.state_dict().items() if '.lora.' not in n}
        for name, module in model.named_modules():
            if hasattr(module, 'lora'):
                merged[name+'.weight'] += module.lora.B.weight.detach().cpu() @ module.lora.A.weight.detach().cpu()
        merged = {n: v.half() for n,v in merged.items()}
        assert all(torch.isfinite(v).all() for v in merged.values())
        # Strict-load the merged checkpoint on CPU before publication.
        probe = MiniMindForCausalLM(MiniMindConfig(hidden_size=768,num_hidden_layers=8)); probe.load_state_dict(merged,strict=True); del probe
        atomic_save(merged, ROOT/'out/merged_medical_768.pth')
        after.update(base_unchanged=True, step=total, optimizer_updates=math.ceil(total/2),
                     adapter_sha256=sha(ROOT/'out/lora_medical_768.pth'),
                     merged_sha256=sha(ROOT/'out/merged_medical_768.pth'), wandb_id=run.id)
        write('evaluation_after.json', after)
        run.summary.update(dict(final_validation_loss=after['loss'], baseline_validation_loss=before['loss'], base_unchanged=True, total_steps=total))
        artifact=wandb.Artifact('chatdoctor-lora-768', type='model', metadata=after)
        artifact.add_file(str(ROOT/'out/lora_medical_768.pth')); artifact.add_file(str(ROOT/'training_settings.json'))
        run.log_artifact(artifact)
    phase('packaging')
    package = RUNTIME/'artifacts'; package.mkdir(exist_ok=True)
    for name in ['dataset_manifest.json','training_settings.json','evaluation_before.json','evaluation_after.json','smoke.json','wandb_run.json','config.json','checkpoint.json','requirements-resolved.txt','README.md']:
        shutil.copyfile(ROOT/name,package/name)
    for path in (ROOT/'out').iterdir(): shutil.copyfile(path,package/path.name)
    shutil.copyfile(ROOT/'checkpoints/lora_medical_768_resume.pth',package/'lora_medical_768_resume.pth')
    shutil.copytree(ROOT/'tokenizer',package/'tokenizer',dirs_exist_ok=True)
    shutil.copytree(RUNTIME/'repo/model',package/'model',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(__file__,package/'run.py')
    hashes = {str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*')) if p.is_file()}
    (package/'SHA256.json').write_text(json.dumps(hashes,indent=2))
    with tarfile.open(RUNTIME/'lora-artifacts.tar.gz','w:gz',compresslevel=1) as archive:
        for path in package.iterdir(): archive.add(path,arcname=path.name)
    shutil.copyfile(RUNTIME/'lora-artifacts.tar.gz',ROOT/'lora-artifacts.tar.gz')
    (ROOT/'SUCCESS').write_text('complete\n'); phase('complete')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','smoke','train']); args=parser.parse_args()
    globals()[args.action]()
