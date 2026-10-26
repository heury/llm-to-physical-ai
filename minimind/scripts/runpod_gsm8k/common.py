"""Shared, credential-free GSM8K experiment definitions."""
import hashlib,json,re
from decimal import Decimal,InvalidOperation
from pathlib import Path
ROOT=Path('/shared-volume/minimind-gsm8k-sft-20261024')
BASE=Path('/shared-volume/minimind-ultrachat-sft-20261024')
RUNTIME=Path('/tmp/minimind-gsm8k')
REV='740312add88f781978c0658806c59bc2815b9866'
LENGTH=340
SUFFIX='\nSolve step by step. End your answer with #### followed by the final number.'
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False));temp.replace(path)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def number(text):
 try:
  result=Decimal(text.replace(',','').strip())
  return str(result.normalize()) if result.is_finite() else None
 except InvalidOperation:return None
NUM=r'[-+]?(?:\d[\d,]*)(?:\.\d+)?'
def extract(text,strict=True):
 # Strict: a valid numeric final line beginning with ####. No gold-dependent extraction.
 marked=re.findall(r'(?m)^\s*####\s*('+NUM+r')\s*$',text)
 if marked:return number(marked[-1])
 if strict:return None
 values=re.findall(NUM,text)
 return number(values[-1]) if values else None

def messages(question,answer=None):
 out=[{'role':'user','content':question+SUFFIX}]
 if answer is not None:out.append({'role':'assistant','content':answer})
 return out

def windows(tokens,labels):
 """340-token windows, 84-token overlap; each shifted assistant target exactly once."""
 result=[];covered=1
 for start in range(0,len(tokens),256):
  end=min(start+LENGTH,len(tokens));x=tokens[start:end];y=labels[start:end].copy()
  for j in range(len(y)):
   if start+j<covered or j==0:y[j]=-100
  covered=max(covered,end)
  if any(v!=-100 for v in y[1:]):
   result.append((x+[0]*(LENGTH-len(x)),y+[-100]*(LENGTH-len(y))))
  if end==len(tokens):break
 assert sum(sum(v!=-100 for v in y[1:]) for x,y in result)==sum(v!=-100 for v in labels[1:])
 return result
