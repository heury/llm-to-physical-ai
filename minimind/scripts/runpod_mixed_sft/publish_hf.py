"""Publish verified final mixed-SFT checkpoints; never publish a partial epoch."""
import argparse, hashlib, importlib.util, json, os, runpy, shutil, subprocess, sys, tarfile, time
from pathlib import Path
import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / 'outputs/minimind/mixed-sft-20261024'
STAGE = REPO / 'outputs/minimind/hf-release-v1.0'
NAMES = {'dense': 'MiniMind-Dense-64M-Mixed-SFT-v1.0', 'moe': 'MiniMind-MoE-198M-A64M-Mixed-SFT-v1.0'}
VERSION = 'v1.0'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)

def extract(archive):
    dest = STAGE / 'inputs'
    dest.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter='data')

def capture_moe():
    """Capture immutable final weights during evaluation, before training supervisor deletes Pod."""
    ssh = runpy.run_path('/tmp/mixed-sft-remote.py')['ssh']
    remote = '/shared-volume/minimind-mixed-sft-20261024'
    command = ('python3 -c ' + __import__('shlex').quote(
        'import json; from pathlib import Path; r=Path(' + repr(remote) + '); '
        's=json.loads((r/"moe/status.json").read_text()); '
        'assert s["phase"] in ["tests_after", "complete_awaiting_export"]; '
        'p=r/"moe/out/full_sft_768_moe.pth"; assert p.exists(); print(p.stat().st_size)'))
    probe = ssh('moe', command, capture_output=True, text=True, timeout=30)
    if probe.returncode:
        return False
    temp = STAGE / 'moe-inputs.tar.tmp'
    with temp.open('wb') as f:
        result = ssh('moe', f'tar -C {remote} -cf - moe/out/full_sft_768_moe.pth moe/config.json', stdout=f, stderr=subprocess.PIPE, timeout=240)
    result.check_returncode()
    temp.replace(STAGE / 'moe-inputs.tar')
    extract(STAGE / 'moe-inputs.tar')
    return True

def package(kind, done):
    weights = STAGE / 'inputs' / kind / 'out' / ('full_sft_768' + ('_moe' if kind == 'moe' else '') + '.pth')
    assert sha(weights) == done['weights_sha256'], 'Final checkpoint SHA mismatch'
    assert done['complete'] and done['tests_complete'] and done['checkpoint_verified']
    assert done['epochs'] == 2 and done['optimizer_updates'] == 84320
    spec = importlib.util.spec_from_file_location('model_minimind', STAGE / 'model_minimind.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['model_minimind'] = module
    spec.loader.exec_module(module)
    config = module.MiniMindConfig(**json.loads((STAGE / 'inputs' / kind / 'config.json').read_text()))
    config.model_version = VERSION
    config.finetuning_task = 'mixed-sft'
    model = module.MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    folder = STAGE / NAMES[kind]
    folder.mkdir(exist_ok=True)
    module.MiniMindConfig.register_for_auto_class()
    module.MiniMindForCausalLM.register_for_auto_class('AutoModelForCausalLM')
    model.half().save_pretrained(folder, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(STAGE / 'inputs/tokenizer', local_files_only=True)
    tok.save_pretrained(folder)
    # Reload the packaged remote-code format and compare to the source checkpoint.
    restored = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True, local_files_only=True, torch_dtype=torch.float32).eval()
    # Rebuild original FP32 RoPE buffers instead of half→float rounding them.
    model = module.MiniMindForCausalLM(config).eval()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    x = torch.tensor([[1, 20, 21, 2]])
    with torch.inference_mode():
        torch.testing.assert_close(model(x).logits, restored(x).logits, rtol=1e-5, atol=1e-6)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        assert (folder / name).exists()
    shutil.copyfile(REPO / 'minimind/LICENSE', folder / 'LICENSE')
    (folder / 'requirements.txt').write_text('torch>=2.6,<3\ntransformers==4.57.6\nsafetensors>=0.4\n')
    metrics = {k: done[k] for k in ['epochs','optimizer_updates','training_seconds','validation_before','validation_after','gsm8k_test_rows','strict_correct_before','strict_correct_after','flexible_correct_before','flexible_correct_after']}
    write(folder / 'evaluation.json', metrics)
    card = f"""---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- minimind
- {'moe' if kind == 'moe' else 'dense'}
- conversational
- experimental
datasets:
- HuggingFaceFW/fineweb-edu
- HuggingFaceH4/ultrachat_200k
- HuggingFaceTB/smoltalk
- openai/gsm8k
---

# {NAMES[kind]}

Version: **{VERSION}**. Final 2-epoch mixed SFT checkpoint, released 2026-10-25.
Architecture: MiniMind {'MoE, 198,416,640 total parameters; 4 experts, top-1 routing (~64M active)' if kind == 'moe' else 'Dense, 63,912,192 parameters'}, hidden size 768, 8 layers, vocabulary 6400.
FP16 safetensors; tied embeddings. This is an experimental English model, not a reliable reasoning assistant.

## Training

Full-document pretraining on FineWeb-Edu sample/10BT filtered by **score >= 3.5**, 2 epochs.
340-token windows, stride 339; incomplete final pretraining window per document dropped.
Mixed SFT: UltraChat train_sft (207,865 conversations), SmolTalk all/train (1,020,912), GSM8K main/train (6,851), after deduplication and held-out prompt exclusion.
Total 1,235,628 conversations / 5,396,469 windows per epoch. Natural mixture, no oversampling.
SFT length 340, stride 256; overlapping labels masked; final short windows padded so assistant targets and EOS are retained.
Assistant-only loss; 2 epochs, 84,320 optimizer updates; BF16; effective batch 128; peak LR 5e-5.
Exact held-out prompt and GSM8K-question substring checks were applied; paraphrase-level or pretraining decontamination is not guaranteed.
Each source's license/terms still apply; Apache-2.0 identifies this model/code release, not a relicensing of source datasets.
Architecture code derived from https://github.com/jingyaogong/minimind (Apache-2.0).

## Evaluation and limitations

Assistant-only validation PPL: {done['validation_before']['perplexity']:.4f} → {done['validation_after']['perplexity']:.4f} (512 fixed windows per source).
GSM8K main/test, all 1,319 questions, zero-shot greedy: **{done['strict_correct_after']}/1319 ({100*done['strict_correct_after']/1319:.2f}%)** with strict final `#### number` scoring.
Last-number fallback: {done['flexible_correct_after']}/1319. See evaluation.json for before/after metrics.
Evaluation uses maximum total context 340 and up to 256 new tokens. Short context/output limits can truncate solutions.
Outputs can contain repetition, factual errors, incorrect arithmetic and failure to follow instructions. Low PPL does not demonstrate reasoning accuracy.
The architectural position buffer permits longer contexts, but **training used length 340**; longer context quality is unvalidated.

## Usage

Install requirements.txt. Review model_minimind.py before enabling custom code.
The prompt template must remove the empty thinking block to match SFT preprocessing.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
repo = "caspar/{NAMES[kind]}"
tokenizer = AutoTokenizer.from_pretrained(repo, revision="v1.0")
model = AutoModelForCausalLM.from_pretrained(
    repo, revision="v1.0", trust_remote_code=True,
    torch_dtype=torch.float32,
).eval()
text = tokenizer.apply_chat_template(
    [{{"role": "user", "content": "Why is the sky blue?"}}],
    tokenize=False, add_generation_prompt=True, open_thinking=False,
).replace('<think>\n\n</think>\n\n', '')
x = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
assert x.shape[1] < 340
with torch.inference_mode():
    y = model.generate(inputs=x, max_new_tokens=min(128, 340-x.shape[1]),
        do_sample=False, temperature=1., top_p=1., top_k=0,
        repetition_penalty=1., eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(y[0, x.shape[1]:], skip_special_tokens=True))
```

The repo contains inference weights, configuration, tokenizer, custom model code, license, minimal dependency list and evaluation summary only. No optimizer states, training datasets, W&B logs or credentials.
"""
    (folder / 'README.md').write_text(card)
    write(folder / 'release.json', {'model_name':NAMES[kind], 'version':VERSION, 'source_checkpoint_sha256':done['weights_sha256'], 'source_code_sha256':sha(STAGE/'model_minimind.py'), 'files':{p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='release.json'}})
    write(STAGE / f'{kind}-package-verified.json', {'strict_reload':True,'weights_equal':True,'logits_equal':True,'version':VERSION})
    return folder

def publish(kind):
    receipt = STAGE / f'{kind}-published.json'
    if receipt.exists():
        return
    done = json.loads((RESULTS / kind / 'completion.json').read_text())
    folder = package(kind, done)
    api = HfApi()
    assert api.whoami()['name'] == 'caspar'
    repo_id = 'caspar/' + NAMES[kind]
    # Existing repos can only be resumed if created by this job.
    marker = STAGE / f'{kind}-repo-created.json'
    if not marker.exists():
        api.create_repo(repo_id, private=False, exist_ok=False)
        write(marker, {'repo_id':repo_id})
    commit = api.upload_folder(repo_id=repo_id, folder_path=folder, commit_message='Release v1.0: verified final 2-epoch mixed SFT')
    api.create_tag(repo_id, tag=VERSION, revision=commit.oid, exist_ok=True)
    info = HfApi(token=False).model_info(repo_id, revision=VERSION, files_metadata=True)
    assert not info.private and info.sha == commit.oid
    # Verify every uploaded payload hash through anonymous, pinned downloads.
    path = Path(snapshot_download(repo_id, revision=commit.oid, token=False, local_dir=STAGE/f'{kind}-public-verification'))
    for file in folder.iterdir():
        if file.is_file():
            assert sha(file) == sha(path / file.name), file.name
    write(receipt, {'repo_id':repo_id,'url':'https://huggingface.co/'+repo_id,'version':VERSION,'commit':commit.oid,'public':True,'anonymous_download_hashes_verified':True,'timestamp':time.time()})
    print(json.dumps(json.loads(receipt.read_text())), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['dense','moe'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    STAGE.mkdir(exist_ok=True)
    if args.kind == 'dense':
        extract(STAGE/'dense-inputs.tar')
    else:
        while not (STAGE/'inputs/moe/out/full_sft_768_moe.pth').exists():
            try:
                if capture_moe(): break
            except Exception as exc:
                print(type(exc).__name__, str(exc)[:200], flush=True)
            write(STAGE/'moe-publish-status.json', {'phase':'waiting_for_final_weights','timestamp':time.time()})
            time.sleep(20)
        while not (RESULTS/'moe/completion.json').exists():
            write(STAGE/'moe-publish-status.json', {'phase':'waiting_for_completed_evaluation','timestamp':time.time()})
            time.sleep(20)
    publish(args.kind)

if __name__ == '__main__':
    main()
