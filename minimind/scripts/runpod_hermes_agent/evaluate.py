import json
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from trainer.train_agent import TOOLS,rollout_single,parse_tool_calls,validate_gt_in_text
from trainer.rollout_engine import SGLangRolloutEngine

class GreedyEngine(SGLangRolloutEngine):
    def rollout(self,*args,**kwargs):
        kwargs['temperature']=0.0
        return super().rollout(*args,**kwargs)


def evaluate(weight,output):
    cfg=MiniMindConfig.from_pretrained(ROOT/'base')
    model=MiniMindForCausalLM(cfg)
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
    engine=GreedyEngine('http://127.0.0.1:8998',str(ROOT/'sglang_policy'),str(ROOT/'eval_policy'))
    engine.update_policy(model)
    del model
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'tokenizer')
    cases=[]
    for expr,answer in [('37*19','703'),('144/12+7','19'),('2**10-24','1000'),('(83-29)*6','324'),('17*23+9','400'),('81**0.5','9')]:
        cases.append(('math',f'Use calculate_math to evaluate {expr}. Report the result.', ['calculate_math'],[answer]))
    cases.extend([
        ('weather','Use the weather tool to check the weather in Beijing.',['get_current_weather'],['28°C','Clear']),
        ('weather','Check the weather in London using the provided tool.',['get_current_weather'],['5°C','Light rain']),
        ('time','Use the time tool to get the current time in Asia/Tokyo.',['get_current_time'],['2025-03-07 15:30:00']),
        ('exchange','Use the exchange-rate tool to obtain the USD to EUR rate.',['get_exchange_rate'],['0.92']),
        ('units','Use the unit converter to convert 0 celsius to fahrenheit.',['unit_converter'],['32']),
        ('units','Use the unit converter to convert 212 fahrenheit to celsius.',['unit_converter'],['100']),
        ('units','Use the unit converter to convert 10 km to miles.',['unit_converter'],['6.2137']),
        ('general','Explain why the sky appears blue in two sentences.',[],[]),
        ('general','Write a polite email asking to reschedule a meeting.',[],[]),
        ('general','Give three practical tips for learning Python.',[],[]),
    ])
    train_questions={r['conversations'][-2].get('content','') for l in (ROOT/'dataset/agent_rl_en.jsonl').read_text().split('\n') if l for r in [json.loads(l)] if len(r['conversations'])>=2}
    samples=[]
    for category,prompt,names,gt in cases:
        tools=[t for t in TOOLS if t['function']['name'] in names] or None
        result=rollout_single(engine,tokenizer,[{'role':'user','content':prompt}],tools,max_turns=3,max_new_tokens=256,thinking_ratio=0,device='cpu')
        answer,context,_,_,_,_,turns,unfinished=result
        calls=[c for turn in turns for c in parse_tool_calls(turn)]
        correct=bool(gt) and not unfinished and len(validate_gt_in_text(answer,gt))==len(gt)
        samples.append({'category':category,'prompt':prompt,'expected':gt,'answer':answer,'turns':turns,'calls':calls,
            'unfinished':unfinished,'exact_prompt_in_training':prompt in train_questions,
            'ground_truth_match':correct if gt else None,
            'expected_tool_called':any(c.get('name') in names for c in calls) if names else None})
    scored=[s for s in samples if s['expected']]
    report={'samples':samples,'scored_samples':len(scored),'ground_truth_matches':sum(s['ground_truth_match'] for s in scored),
        'expected_tool_called':sum(s['expected_tool_called'] for s in scored),'greedy':True,'max_new_tokens_per_turn':256,
        'max_turns':3,'limitation':'Small synthetic sample check, not a general benchmark; exact prompt overlap checked, paraphrase overlap not excluded.'}
    Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}),flush=True)
    return report


if __name__=='__main__':evaluate(sys.argv[1],sys.argv[2])
