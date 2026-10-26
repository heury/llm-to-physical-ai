"""Real-tokenizer native-label parity, EOS/window and pretrained GPU SFT checks."""
import gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dataset.lm_dataset import SFTDataset
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from common import *
torch.set_num_threads(4);tok=AutoTokenizer.from_pretrained(BASE/'tokenizer',local_files_only=True)
bos=tok(tok.bos_token+'assistant\n',add_special_tokens=False).input_ids;eos=tok(tok.eos_token+'\n',add_special_tokens=False).input_ids
native=SFTDataset.__new__(SFTDataset);native.bos_id=bos;native.eos_id=eos
examples=[messages('What is 2 + 3?','2 + 3 = 5.\n#### 5'),[{'role':'system','content':'Answer briefly.'},{'role':'user','content':'Explain sunlight.'},{'role':'assistant','content':'Plants use sunlight to make food. '*180},{'role':'user','content':'Give a short summary.'},{'role':'assistant','content':'Plants need light.'}],[{'role':'user','content':'Long context. '*400},{'role':'assistant','content':'A short final answer.'}]]
pairs=[];checks=[]
for msg in examples:
 ids=tok(chat_text(tok,msg),add_special_tokens=False).input_ids;a=np.array(ids,dtype=np.int16);y=np.full(len(ids),-100,dtype=np.int16)
 for b,e in assistant_spans(ids,bos,eos):y[b:e]=a[b:e]
 native.max_length=len(ids);assert y.tolist()==native.generate_labels(ids)
 expected=int((y[1:]!=-100).sum());expected_eos=int((y[1:]==2).sum());covered=1;targets=eos_count=0
 for b,e in window_bounds(len(ids)):
  lab=y[b:e].copy();lab[:max(1,covered-b)]=-100;covered=max(covered,e)
  if not (lab[1:]!=-100).any():continue
  pair=np.empty((2,340),dtype=np.int16);pair[0].fill(0);pair[1].fill(-100);pair[0,:e-b]=a[b:e];pair[1,:e-b]=lab;pairs.append(pair)
  targets+=int((lab[1:]!=-100).sum());eos_count+=int((lab[1:]==2).sum())
 assert targets==expected and eos_count==expected_eos
 checks.append(dict(tokens=len(ids),targets=targets,eos_targets=eos_count,native_mask_equal=True))
rows=[]
for kind in ['dense','moe']:
 c=json.loads((BASE/kind/'completion.json').read_text());assert sha(c['weights'])==c['weights_sha256']
 cfg=MiniMindConfig(**json.loads((BASE/kind/'config.json').read_text()));model=MiniMindForCausalLM(cfg).cuda().train();model.load_state_dict(torch.load(c['weights'],map_location='cpu',weights_only=True),strict=True)
 opt=torch.optim.AdamW(model.parameters(),lr=5e-5);batch=np.stack([pairs[i%len(pairs)] for i in range(64)]).astype(np.int64);x=torch.from_numpy(batch[:,0]).cuda();y=torch.from_numpy(batch[:,1]).cuda()
 with torch.autocast('cuda',dtype=torch.bfloat16):res=model(x,labels=y);loss=res.loss+res.aux_loss
 assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
 if kind=='moe':
  gates=[p.grad for n,p in model.named_parameters() if n.endswith('mlp.gate.weight')];assert len(gates)==8 and all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in gates)
 opt.step();rows.append(dict(model=kind,strict_pretrain_reload=True,loss=float(loss.detach()),grad_norm=float(norm),batch_size=64,max_seq_len=340,source_sha256=c['weights_sha256']))
 del model,opt,x,y,res,loss;gc.collect();torch.cuda.empty_cache()
write(ROOT/'smoke.json',dict(passed=True,native_and_eos_checks=checks,models=rows,smoke_weights_discarded=True,timestamp=time.time()));print(json.dumps(rows),flush=True)
