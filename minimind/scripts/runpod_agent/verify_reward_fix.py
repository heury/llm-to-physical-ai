"""Local controlled math-tool warmup; no cloud calls or publication.

This checks whether the base policy can learn valid calls on held-out operands.
It is not an end-to-end RL benchmark or a replacement for the original dataset.
"""
import json
import random
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.train_agent import TOOLS, parse_tool_calls, execute_tool, calculate_rewards

OUT = ROOT.parent / 'outputs/minimind/agent-rl-20261026/reward_fix'
BASE = Path('/tmp/agent-base-model')
TOOLS_MATH = [t for t in TOOLS if t['function']['name'] == 'calculate_math']

def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    tokenizer.chat_template = "{% filter replace('<think>\\n\\n</think>\\n\\n', '') %}" + tokenizer.chat_template + '{% endfilter %}'
    model = MiniMindForCausalLM(MiniMindConfig.from_pretrained(BASE))
    weights = load_file(BASE / 'model.safetensors')
    weights.setdefault('lm_head.weight', weights['model.embed_tokens.weight'])
    model.load_state_dict(weights, strict=True)
    del weights
    def prompt(expr):
        text = tokenizer.apply_chat_template([{'role':'user','content':f'Use calculate_math to evaluate {expr}. Report the result.'}],tools=TOOLS_MATH,tokenize=False,add_generation_prompt=True,open_thinking=False)
        return tokenizer(text,add_special_tokens=False)['input_ids']
    test = ['37*19','144/12+7','2**10-24','(83-29)*6','17*23+9','81**0.5','41+58','96/8']
    def evaluate():
        model.eval()
        result=[]
        for expr in test:
            ids=torch.tensor([prompt(expr)])
            with torch.no_grad():
                y=model.generate(ids,max_new_tokens=96,do_sample=False,eos_token_id=tokenizer.eos_token_id)
            text=tokenizer.decode(y[0,ids.shape[1]:],skip_special_tokens=True)
            calls=parse_tool_calls(text)
            expected=execute_tool('calculate_math',{'expression':expr})
            actual=execute_tool(calls[0]['name'],calls[0].get('arguments',{})) if len(calls)==1 else None
            result.append({'expression':expr,'output':text,'valid_single_call':len(calls)==1 and actual is not None,'correct_execution':actual==expected})
        return result
    before=evaluate()
    print(json.dumps({'stage':'before','correct':sum(x['correct_execution'] for x in before),'total':len(test)}),flush=True)
    examples=[]
    expressions=[]
    for _ in range(96):
        a,b,c=random.randint(11,90),random.randint(2,20),random.randint(1,15)
        expressions.append(random.choice([f'{a}*{b}',f'{a}+{b}',f'{a}-{b}',f'{a}*{b}+{c}',f'({a}-{b})*{c}',f'{a}/{b}',f'{b}**2-{c}']))
    expressions=[e for e in expressions if e not in test]
    for expr in expressions:
        p=prompt(expr)
        target='<tool_call>\n'+json.dumps({'name':'calculate_math','arguments':{'expression':expr}})+'\n</tool_call>'
        target_ids=tokenizer(target,add_special_tokens=False)['input_ids']+[tokenizer.eos_token_id]
        examples.append((p+target_ids,[-100]*len(p)+target_ids))
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
    start=time.time();losses=[]
    model.train()
    for step in range(120):
        batch=random.sample(examples,2);length=max(len(x[0]) for x in batch)
        ids=torch.tensor([x+[tokenizer.pad_token_id]*(length-len(x)) for x,_ in batch])
        labels=torch.tensor([y+[-100]*(length-len(y)) for _,y in batch])
        mask=torch.tensor([[1]*len(x)+[0]*(length-len(x)) for x,_ in batch])
        logits=model(ids,attention_mask=mask).logits
        loss=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100)
        assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);optimizer.step()
        losses.append(loss.item())
        if (step+1)%20==0:print(json.dumps({'stage':'warmup','step':step+1,'loss':loss.item(),'seconds':time.time()-start}),flush=True)
    after=evaluate()
    torch.save({k:v.half() for k,v in model.state_dict().items()},OUT/'math_warmup_768.pth')
    report={'base':str(BASE),'seed':42,'steps':120,'batch_size':2,'learning_rate':2e-5,'training_expressions':expressions,'heldout_expressions':test,'before':before,'after':after,'loss_first20':sum(losses[:20])/20,'loss_last20':sum(losses[-20:])/20,'seconds':time.time()-start,'limitation':'Small same-template held-out-operand SFT experiment. No full RL retraining; no final natural-language answer evaluation or other tool evaluation.'}
    (OUT/'warmup_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'stage':'done','before_correct':sum(x['correct_execution'] for x in before),'after_correct':sum(x['correct_execution'] for x in after),'total':len(test)}),flush=True)

if __name__=='__main__':main()
