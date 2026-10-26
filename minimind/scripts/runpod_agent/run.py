"""Run actual train_agent.py, evaluate before/after, and export a verified model."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def save(name,value):
    path=ROOT/name;tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def train(smoke=False,batch_size=4):
    weight='agent_smoke' if smoke else 'agent_en'
    command=[sys.executable,'-u','train_agent.py','--epochs','1','--batch_size',str(batch_size),
        '--learning_rate','3e-7','--num_generations','4','--max_gen_len','64' if smoke else '256',
        '--max_total_len','2500','--thinking_ratio','0','--rollout_engine','sglang','--rollout_workers',str(batch_size*4),
        '--train_microbatch_size','1','--sglang_base_url','http://127.0.0.1:8998','--sglang_model_path','../sglang_policy',
        '--sglang_shared_path','../policy_updates','--tokenizer_path','../tokenizer',
        '--reward_model_path','../reward_skywork','--reward_score_scale','10','--from_weight','full_sft','--save_weight',weight,
        '--save_interval','200','--policy_update_interval','10','--num_workers','2',
        '--data_path','../dataset/agent_smoke.jsonl' if smoke else '../dataset/agent_rl_en.jsonl']
    if not smoke:command+=['--use_wandb','--wandb_mode','offline','--wandb_project','MiniMind-Agent-RL']
    save('training_command.json' if not smoke else 'smoke_command.json',command)
    start=time.time();log=ROOT/('smoke.log' if smoke else 'train.log')
    with log.open('w') as f:
        process=subprocess.Popen(command,cwd=ROOT/'trainer',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            f.write(line);f.flush();print(line,end='',flush=True)
            match=re.search(r'Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\), Reward:([^,]+)',line)
            if match:
                step=int(match[3]);total=int(match[4]);elapsed=time.time()-start
                save('status.json',{'stage':'smoke' if smoke else 'training','step':step,'total_steps':total,
                    'reward':float(match[5]),'elapsed_seconds':elapsed,'seconds_per_step':elapsed/step,
                    'estimated_remaining_seconds':elapsed/step*(total-step),'updated_at':time.time()})
        code=process.wait()
    if code:raise RuntimeError(f'{weight} training failed with exit {code}; see {log}')
    return time.time()-start


def main():
    import torch
    from transformers import AutoTokenizer
    from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
    from trainer.rollout_engine import SGLangRolloutEngine,compute_per_token_logps
    from trainer.trainer_utils import LMForRewardModel
    from evaluate import evaluate
    torch.set_num_threads(4)
    assert torch.cuda.get_device_name().find('5090')>=0
    save('status.json',{'stage':'preflight','updated_at':time.time()})
    cfg=MiniMindConfig.from_pretrained(ROOT/'base')
    model=MiniMindForCausalLM(cfg).cuda().eval()
    model.load_state_dict(torch.load(ROOT/'out/full_sft_768.pth',map_location='cpu',weights_only=True),strict=True)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'tokenizer')
    engine=SGLangRolloutEngine('http://127.0.0.1:8998',str(ROOT/'sglang_policy'),str(ROOT/'preflight_policy'))
    engine.update_policy(model)
    text=tokenizer.apply_chat_template([{'role':'user','content':'Hello! Tell me about yourself.'}],tokenize=False,add_generation_prompt=True,open_thinking=False)
    enc=tokenizer(text,return_tensors='pt',add_special_tokens=False).to('cuda')
    result=engine.rollout(enc.input_ids,enc.attention_mask,1,8,temperature=1.0)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        lp=compute_per_token_logps(model,result.output_ids,result.completion_ids.size(1))
    difference=(lp-result.per_token_logps).abs().max().item()
    assert difference < .5, f'SGLang/native token logprob difference: {difference}'
    save('preflight.json',{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'sglang_native_logprob_max_difference':difference,'generated_ids':result.completion_ids.tolist(),'weight_reload_verified':True})
    del model,lp,result;torch.cuda.empty_cache()
    reward=LMForRewardModel(str(ROOT/'reward_skywork'),device='cuda',dtype=torch.bfloat16,score_scale=10.0)
    score=reward.get_score([{'role':'user','content':'What is 2+2?'}],'The answer is 4.')
    assert math.isfinite(score)
    good_raw=reward.last_raw_score
    bad_score=reward.get_score([{'role':'user','content':'What is 2+2?'}],'The answer is 17.')
    save('reward_preflight.json',{'good_score':score,'bad_score':bad_score,'good_raw':good_raw,'bad_raw':reward.last_raw_score,
        'model':'Skywork/Skywork-Reward-V2-Qwen3-1.7B','revision':'e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc','score_scale':10.0})
    del reward;torch.cuda.empty_cache()
    save('status.json',{'stage':'baseline_evaluation','updated_at':time.time()})
    evaluate(ROOT/'out/full_sft_768.pth',ROOT/'evaluation_before.json')
    smoke_seconds=train(smoke=True)
    save('smoke_complete.json',{'seconds':smoke_seconds,'batch_size':4})
    training_seconds=train()
    final_step=re.findall(r'Epoch:\[1/1\]\((\d+)/(\d+)\)',(ROOT/'train.log').read_text())
    assert len(final_step)==math.ceil(30196/4) and final_step[-1]==('7549','7549')
    state=torch.load(ROOT/'out/agent_en_768.pth',map_location='cpu',weights_only=True)
    assert all(torch.isfinite(t).all() for t in state.values())
    model=MiniMindForCausalLM(cfg);model.load_state_dict(state,strict=True)
    base_state=torch.load(ROOT/'out/full_sft_768.pth',map_location='cpu',weights_only=True)
    assert any(not torch.equal(t,base_state[k]) for k,t in state.items())
    del base_state
    save('status.json',{'stage':'final_evaluation','updated_at':time.time()})
    after = evaluate(ROOT/'out/agent_en_768.pth',ROOT/'evaluation_after.json')
    before = json.loads((ROOT/'evaluation_before.json').read_text())
    if after['ground_truth_matches'] <= before['ground_truth_matches'] or after['expected_tool_called'] == 0:
        raise RuntimeError('Publication blocked: sample tool accuracy did not improve over the base model; checkpoint retained')
    model.config.finetuning_task='agent-rl';model.config.model_version='v1.0'
    model=model.half().eval();export=ROOT/'export';export.mkdir(exist_ok=True)
    MiniMindConfig.register_for_auto_class();MiniMindForCausalLM.register_for_auto_class('AutoModelForCausalLM')
    model.save_pretrained(export,safe_serialization=True);tokenizer.save_pretrained(export)
    from transformers import AutoModelForCausalLM
    reloaded=AutoModelForCausalLM.from_pretrained(export,trust_remote_code=True,torch_dtype=torch.float16)
    assert all(torch.equal(v,reloaded.state_dict()[k]) for k,v in model.state_dict().items())
    summary={'complete':True,'epochs':1,'training_rows':30196,'batch_size':4,'optimizer_updates':math.ceil(30196/4),
        'training_seconds':training_seconds,'finite_weights':True,'strict_reload_verified':True,
        'native_checkpoint_sha256':hashlib.sha256((ROOT/'out/agent_en_768.pth').read_bytes()).hexdigest(),
        'model_sha256':hashlib.sha256((export/'model.safetensors').read_bytes()).hexdigest()}
    save('completion.json',summary);save('status.json',{'stage':'ready_to_publish',**summary,'updated_at':time.time()})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save('status.json',{'stage':'failed','error':str(exc),'traceback':traceback.format_exc(),'updated_at':time.time()})
        raise
