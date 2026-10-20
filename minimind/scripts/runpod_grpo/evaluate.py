import hashlib,json,os,statistics
from pathlib import Path
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from reward_adapter import BatchedRewardModel
from trainer.train_grpo import rep_penalty
root=Path(os.environ['GRPO_ROOT']);runtime=Path(os.environ['GRPO_RUNTIME']);phase=os.environ.get('GRPO_PHASE','before')
torch.set_num_threads(4);torch.manual_seed(42)
tok=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True)
weight=Path('/shared-volume/minimind-dpo-ultrafeedback-20261020/out/dpo_768.pth') if phase=='before' else root/'out/grpo_768.pth'
state=torch.load(weight,map_location='cpu',weights_only=True);assert all(torch.isfinite(t).all() for t in state.values())
model=MiniMindForCausalLM(MiniMindConfig(hidden_size=768,num_hidden_layers=8,max_seq_len=1024));model.load_state_dict(state,strict=True);del state
model=model.cuda().eval();reward=BatchedRewardModel(runtime/'reward_model')
with (runtime/'validation.jsonl').open(encoding='utf-8') as f:rows=[json.loads(line) for line in f]
records=[];rewards=[];lengths=[];penalties=[];eos_flags=[]
@torch.inference_mode()
def generate(messages):
    texts=[tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True,open_thinking=False) for m in messages]
    inputs=tok(texts,return_tensors='pt',padding=True,padding_side='left',add_special_tokens=False).to('cuda')
    assert inputs.input_ids.shape[1]<=768
    with torch.autocast('cuda',dtype=torch.bfloat16):
        ids=model.generate(inputs=inputs.input_ids,attention_mask=inputs.attention_mask,max_new_tokens=256,do_sample=False,temperature=1.,top_p=1.,top_k=0,logits_to_keep=1,eos_token_id=tok.eos_token_id)
    comp=ids[:,inputs.input_ids.shape[1]:]
    mask=(comp.eq(tok.eos_token_id).cumsum(1)-comp.eq(tok.eos_token_id).long()).eq(0)
    responses=tok.batch_decode(comp,skip_special_tokens=True)
    raw=reward.score_batch(messages,responses).tolist()
    for msg,response,score,n,eos in zip(messages,responses,raw,mask.sum(1).tolist(),comp.eq(tok.eos_token_id).any(1).tolist()):
        penalty=rep_penalty(response);value=max(-20,min(20,score))/5-penalty if response.strip() else -5.
        records.append(dict(messages=msg,response=response,raw_reward=score,reward=value,length=n,eos=eos,repetition_penalty=penalty))
for start in range(0,len(rows),8):generate([r['conversations'][:-1] for r in rows[start:start+8]])
validation=records[:];records=[]
for prompt in ['Explain photosynthesis in two sentences.','How do I calculate the area of a circle?','Write a Python function that adds two numbers.','List three tips for learning a language.']:
    generate([[dict(role='system',content='You are a helpful assistant.'),dict(role='user',content=prompt)]])
report=dict(phase=phase,validation_prompts=len(validation),mean_reward=statistics.mean(r['reward'] for r in validation),mean_raw_reward=statistics.mean(r['raw_reward'] for r in validation),mean_repetition_penalty=statistics.mean(r['repetition_penalty'] for r in validation),mean_completion_tokens=statistics.mean(r['length'] for r in validation),eos_rate=statistics.mean(r['eos'] for r in validation),empty_rate=statistics.mean(not r['response'].strip() for r in validation),manual_samples=records,evaluation_type='Fixed128test prompts; greedy decoding; same frozen scorer as training, not independent accuracy benchmark')
if phase=='after':
    c=torch.load(root/'checkpoints/grpo_768_resume.pth',map_location='cpu',weights_only=True)
    manifest=json.loads((root/'dataset_manifest.json').read_text());assert c['epoch']==0 and c['step']==manifest['steps_per_epoch']
    assert c['wandb_id'];report['wandb_id']=c['wandb_id'];report['final_step']=c['step'];report['optimizer_updates']=int(next(iter(c['optimizer']['state'].values()))['step'])
    assert all(torch.isfinite(v).all() for s in c['optimizer']['state'].values() for v in s.values() if torch.is_tensor(v))
    with weight.open('rb') as f:report['checkpoint_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
(root/('evaluation_'+phase+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2))
(root/('evaluation_'+phase+'_samples.json')).write_text(json.dumps(validation,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False),flush=True)
