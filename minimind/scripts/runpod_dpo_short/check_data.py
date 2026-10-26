"""Check length, masks, split leakage, and the DPO gradient direction."""
import json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from prepare import ROOT,render

def main():
 tok=AutoTokenizer.from_pretrained(ROOT/'base',local_files_only=True);groups=[];count=0
 for split in ['train','validation','test']:
  a=np.load(ROOT/'data'/f'{split}.npy');rows=[json.loads(s) for s in (ROOT/'data'/f'{split}.jsonl').read_text().splitlines()];groups.append({r['prompt_sha256'] for r in rows});assert len(a)==len(rows)
  for i,r in enumerate(rows):
   for side,key in enumerate(['chosen','rejected']):
    ids=tok(render(tok,r[key]),add_special_tokens=False).input_ids;n=r['prompt_tokens'];assert len(ids)<340 and len(ids)==r[key+'_tokens']
    assert a[i,side,0,:len(ids)].tolist()==ids and np.all(a[i,side,0,len(ids):]==0)
    assert np.all(a[i,side,1,:n]==-100) and np.all(a[i,side,1,len(ids):]==-100)
    assert a[i,side,1,n:len(ids)].tolist()==ids[n:];assert tok.eos_token_id in ids[n:]
   count+=1
 assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
 p=torch.tensor([[-10.,-12.]],requires_grad=True);r=p.detach().clone();loss=-F.logsigmoid(.1*((p[:,0]-p[:,1])-(r[:,0]-r[:,1]))).mean();loss.backward();assert p.grad[0,0]<0 and p.grad[0,1]>0
 result={'pairs_checked':count,'strict_length_under_340':True,'all_response_masks_and_eos_checked':True,'prompt_disjoint_splits':True,'dpo_gradient_prefers_chosen':True}
 (ROOT/'data_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
