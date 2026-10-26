"""Frozen three-source SFT, full assistant targets and final EOS retained."""
import hashlib,json,re,unicodedata,bisect
from pathlib import Path
from decimal import Decimal,InvalidOperation
ROOT=Path('/shared-volume/minimind-mixed-sft-20261024')
BASE=Path('/shared-volume/minimind-full-pretrain-20261024')
RUNTIME=Path('/tmp/minimind-mixed-sft')
PROJECT='MiniMind-Mixed-SFT'
LENGTH=340
STRIDE=256
SUFFIX='\nSolve step by step. End your answer with #### followed by the final number.'
REVS={'ultrachat':'8049631c405ae6576f93f445c6b8166f76f5505a','smoltalk':'5feaf2fd3ffca7c237fc38d1861bc30365d48ffa','gsm8k':'740312add88f781978c0658806c59bc2815b9866'}
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False));temp.replace(path)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(text):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text)).strip().casefold()
def prompt_key(messages):
 text=next((m['content'] for m in messages if m['role']=='user'),'');return hashlib.sha256(canonical(text).encode()).digest()
def conversation_key(messages):return hashlib.sha256(json.dumps(messages,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).digest()
def messages(question,answer=None):
 result=[{'role':'user','content':question+SUFFIX}]
 if answer is not None:result.append({'role':'assistant','content':answer})
 return result
def chat_text(tok,msgs,generation=False):
 text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=generation,open_thinking=False)
 return text.replace('<think>\n\n</think>\n\n','')
def normalize(row,source):
 if source=='gsm8k':return messages(row['question'],row['answer'])
 original=row.get('messages')
 if not isinstance(original,list) or not original:return None
 result=[]
 for m in original:
  role=m.get('role');content=m.get('content')
  if role not in ['system','user','assistant','tool'] or not isinstance(content,str):return None
  new={'role':role,'content':content}
  for key in ['reasoning_content','tool_calls']:
   if m.get(key):new[key]=m[key]
  result.append(new)
 if not any(m['role']=='user' for m in result) or not any(m['role']=='assistant' for m in result):return None
 return result
def assistant_spans(ids,bos,eos):
 # Find tokenized role/EOS markers; skip nested header literals inside a response,
 # exactly as native SFTDataset.generate_labels scans past that response.
 import numpy as np
 a=np.asarray(ids);starts=np.flatnonzero(a==bos[0]).tolist();ends=[i for i in np.flatnonzero(a==eos[0]).tolist() if ids[i:i+len(eos)]==eos]
 spans=[];cursor=0
 for i in starts:
  if i<cursor or ids[i:i+len(bos)]!=bos:continue
  begin=i+len(bos);j=bisect.bisect_left(ends,begin);end=ends[j] if j<len(ends) else len(ids)
  stop=min(end+len(eos),len(ids));spans.append((begin,stop));cursor=stop
 return spans
def window_bounds(length):
 for start in range(0,length,STRIDE):
  end=min(start+LENGTH,length);yield start,end
  if end==length:break
def number(text):
 try:
  n=Decimal(text.replace(',','').strip());return str(n.normalize()) if n.is_finite() else None
 except InvalidOperation:return None
NUM=r'[-+]?(?:\d[\d,]*)(?:\.\d+)?'
def extract(text,strict=True):
 marked=re.findall(r'(?m)^\s*####\s*('+NUM+r')\s*$',text)
 if marked:return number(marked[-1])
 if strict:return None
 vals=re.findall(NUM,text);return number(vals[-1]) if vals else None

def read_json(path,retries=120,delay=.25):
 """Retry transient partial/missing shared-volume JSON publication reads."""
 import time
 for attempt in range(retries):
  try:return json.loads(Path(path).read_text())
  except (json.JSONDecodeError,FileNotFoundError):
   if attempt+1==retries:raise
   time.sleep(delay)
