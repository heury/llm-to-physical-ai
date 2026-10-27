"""Pin downloads, check native/Qwen3 parity, and stage the Agent RL run."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download, hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

BASE = 'caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0'
BASE_REVISION = '7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d'
DATA = 'caspar/agent-rl-en'
DATA_REVISION = '57e3a151f11b0cdd2e150dc6fb2e508842b96ab0'


def main():
    torch.set_num_threads(4)
    base = Path(snapshot_download(BASE, revision=BASE_REVISION, local_dir=ROOT/'base'))
    source = hf_hub_download(DATA, 'agent_rl_en.jsonl', repo_type='dataset', revision=DATA_REVISION)
    target = ROOT/'dataset/agent_rl_en.jsonl'
    shutil.copyfile(source, target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == 'd93641d906274d2781470551fbd91797f8681b129f29f47f119fed4447b073b3'
    cfg = MiniMindConfig.from_pretrained(base)
    model = MiniMindForCausalLM(cfg)
    state = load_file(base/'model.safetensors')
    state.setdefault('lm_head.weight', state['model.embed_tokens.weight'])
    warmup = ROOT/'warmup_seed.pth'
    if warmup.exists():
        state = torch.load(warmup, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    (ROOT/'out').mkdir(exist_ok=True)
    torch.save(state, ROOT/'out/full_sft_768.pth')
    tok = AutoTokenizer.from_pretrained(base)
    tok.chat_template = "{% filter replace('<think>\\n\\n</think>\\n\\n', '') %}" + tok.chat_template + '{% endfilter %}'
    tok.save_pretrained(ROOT/'tokenizer')
    text = tok.apply_chat_template([{'role':'user','content':'Hello'}],tokenize=False,add_generation_prompt=True,open_thinking=False)
    assert '<think>' not in text
    qcfg = Qwen3Config(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size, num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads, num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps, rope_theta=cfg.rope_theta, tie_word_embeddings=True,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    qmodel = Qwen3ForCausalLM(qcfg)
    qmodel.load_state_dict(state, strict=True)
    model.eval(); qmodel.eval()
    x=tok(text, return_tensors='pt', add_special_tokens=False).input_ids
    with torch.no_grad():
        difference=(model(x).logits-qmodel(x).logits).abs().max().item()
    assert difference < 1e-4, difference
    qmodel.to(torch.bfloat16).save_pretrained(ROOT/'sglang_policy')
    tok.save_pretrained(ROOT/'sglang_policy')
    rows=target.read_text().split('\n'); rows=[r for r in rows if r]
    assert len(rows)==30196
    # Deliberately include general dialogue and all five retained tool types in smoke.
    chosen=[];seen=set()
    for line in rows:
        r=json.loads(line)
        names=tuple(sorted(t['function']['name'] for m in r['conversations'] if m.get('tools') for t in json.loads(m['tools'])))
        if names not in seen:
            seen.add(names);chosen.append(line)
        if len(chosen)==16:break
    (ROOT/'dataset/agent_smoke.jsonl').write_text('\n'.join(chosen)+'\n')
    record={'base':BASE,'base_revision':BASE_REVISION,'dataset':DATA,'dataset_revision':DATA_REVISION,
        'warmup_seed_sha256':hashlib.sha256(warmup.read_bytes()).hexdigest() if warmup.exists() else None,
        'warmup_details':'120 math-call SFT updates, batch 2, LR 2e-5; base-derived seed' if warmup.exists() else None,
        'rows':len(rows),'native_qwen3_max_logit_difference':difference,'parameters':sum(p.numel() for p in model.parameters()),
        'tokenizer_vocab_size':len(tok),'empty_think_removed':True,'torch':torch.__version__}
    (ROOT/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
