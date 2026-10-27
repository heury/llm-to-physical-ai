"""Hermes conversion with conversation-level split and preserved assistant targets."""
import hashlib
import json
import re

REVISION = 'dae3e1d28cfbcf4b915c04ea1e072030529b4bda'
SOURCE = 'NousResearch/hermes-function-calling-v1'
FILE = 'func-calling.json'
SYSTEM = 'You are a helpful assistant. Use the provided tools when needed. After receiving tool results, answer the user.'

def normalize(row):
    tools = row.get('tools') or []
    if isinstance(tools, str): tools = json.loads(tools)
    messages = [{'role':'system','content':SYSTEM}]
    for m in row['conversations']:
        role = {'human':'user','gpt':'assistant'}.get(m['from'],m['from'])
        if role == 'system': continue  # tool schema is rendered exactly once by the tokenizer
        if role not in ('user','assistant','tool'): raise ValueError('Unknown role '+role)
        text=m['value']
        if role == 'tool':
            responses=re.findall(r'<tool_response>(.*?)</tool_response>',text,re.S)
            if responses:
                values=[json.loads(x.strip()) for x in responses]
                text=json.dumps(values[0] if len(values)==1 else values,ensure_ascii=False)
        if role == 'assistant':
            def canonical(match):
                call=json.loads(match[1].strip())
                if not isinstance(call,dict) or not isinstance(call.get('name'),str) or not isinstance(call.get('arguments'),dict):
                    raise ValueError('Malformed tool call')
                return '<tool_call>\n'+json.dumps(call,ensure_ascii=False)+'\n</tool_call>'
            text=re.sub(r'<tool_call>(.*?)</tool_call>',canonical,text,flags=re.S)
        messages.append({'role':role,'content':text})
    group='\n'.join(m['content'].strip().lower() for m in messages if m['role']=='user')
    validation=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%10==0
    return {'id':row['id'],'messages':messages,'tools':tools,'validation':validation}

def examples(row,tokenizer,max_length=4096):
    result=[];skipped=0
    for i,m in enumerate(row['messages']):
        if m['role']!='assistant' or not m['content'].strip():continue
        context=row['messages'][:i]
        prefix=tokenizer.apply_chat_template(context,tools=row['tools'] or None,tokenize=False,add_generation_prompt=True,open_thinking=False)
        full=tokenizer.apply_chat_template(row['messages'][:i+1],tools=row['tools'] or None,tokenize=False,add_generation_prompt=False,open_thinking=False)
        if not full.startswith(prefix):raise ValueError('Assistant template boundary mismatch')
        p=tokenizer(prefix,add_special_tokens=False)['input_ids']
        t=tokenizer(full[len(prefix):],add_special_tokens=False)['input_ids']
        if len(p)+len(t)>max_length:
            skipped+=1;continue  # Never truncate away the final answer/EOS.
        if tokenizer.eos_token_id not in t:raise ValueError('Missing supervised EOS')
        result.append({'id':row['id'],'ids':p+t,'labels':[-100]*len(p)+t,'prompt':p,'target':m['content'],
                       'post_tool':bool(context and context[-1]['role']=='tool'),'tools':row['tools']})
    return result,skipped
