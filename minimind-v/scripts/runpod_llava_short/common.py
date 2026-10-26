"""Pinned LLaVA training job; no credentials are stored in this module."""
import hashlib, json, os, time
from pathlib import Path
ROOT = Path('/shared-volume/minimind-v-llava-short-20261025')
LOCAL = Path('/tmp/minimind-v-llava-short-20261025')
BASE = Path('/shared-volume/minimind-mixed-sft-20261024')
OLD_VISION = Path('/shared-volume/minimind-v-sharegpt4v-20261020/vision_encoder')
LENGTH = 340
OVERLAP = 64
REVISIONS = {'pretrain':'70f9d1e5e1a697fe35830875cfc7de1dd590d727', 'sft':'9d451dc7629cfe0469f6ae4432b765cd603d5fcb', 'vision':'94dffa8cb1179de3e03f091dbc3917e5d5a9ae84'}
BASE_SHA = 'a1d17222e66cf02cd2601d0b06b9cb76a455e185bee1b3dbc3725ca9c364a211'
SETTINGS = dict(pretrain_dataset='liuhaotian/LLaVA-Pretrain', sft_dataset='liuhaotian/LLaVA-Instruct-150K', revisions=REVISIONS, epochs_alignment=1, epochs_sft=1, sequence_length=LENGTH, overlap=OVERLAP, batch_size=64, accumulation_steps=2, effective_batch_size=128, alignment_lr=4e-4, sft_lr=1e-5, seed=42, vision='google/siglip2-base-patch32-256', frozen_vision=True, filter_policy='caption_under20_words_seed42_50k; standalone_visual_QA_answer_1or2_sentences_max40words_no_complex_reasoning', language_base_sha256=BASE_SHA, wandb_project='MiniMind-V-LLaVA-Short')
def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def status(phase, **kwargs):
    d=dict(phase=phase,timestamp=time.time(),**kwargs);write(ROOT/'status.json',d);print(json.dumps(d),flush=True)
def chat(tok,messages,generation=False):
    return tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=generation,open_thinking=False).replace('<think>\n\n</think>\n\n','')
def encode(tok,messages,generation=False):
    return tok(chat(tok,messages,generation),add_special_tokens=False).input_ids
def messages(row):
    out=[]
    for turn in row['conversations']:
        role={'human':'user','gpt':'assistant'}[turn['from']]
        out.append(dict(role=role,content=turn['value'].replace('<image>','<|image_pad|>'*64)))
    assert out[0]['role']=='user'
    return out

def windows(tok, row):
    """Keep image prefix in every window; supervise each assistant token exactly once."""
    msgs=messages(row);full=encode(tok,msgs);labels=[-100]*len(full)
    for j,m in enumerate(msgs):
        if m['role']=='assistant':
            prefix=encode(tok,msgs[:j],True);end=encode(tok,msgs[:j+1])
            assert end[:len(prefix)]==prefix and full[:len(end)]==end
            labels[len(prefix):len(end)]=full[len(prefix):len(end)]
    positions=[i for i,t in enumerate(full) if t==12]
    assert len(positions)==64 and positions==list(range(positions[0],positions[0]+64))
    cut=positions[-1]+1;prefix=full[:cut];tail=full[cut:];target=labels[cut:]
    capacity=LENGTH-cut;assert capacity>OVERLAP+16 and all(x==-100 for x in labels[:cut])
    stride=capacity-OVERLAP;covered=0
    for start in range(0,len(tail),stride):
        x=prefix+tail[start:start+capacity];y=[-100]*cut+target[start:start+capacity]
        if start:y[cut:cut+OVERLAP]=[-100]*min(OVERLAP,len(y)-cut)
        n=sum(v!=-100 for v in y[1:])
        if n:
            covered+=n
            yield x+[tok.pad_token_id]*(LENGTH-len(x)), y+[-100]*(LENGTH-len(y))
        if start+capacity>=len(tail):break
    assert covered==sum(v!=-100 for v in labels[1:])
