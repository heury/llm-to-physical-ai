"""Actual 5090 BF16 forward/backward/AdamW checks; smoke weights are discarded."""
import gc,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from common import ROOT,write
rows=[];torch.set_num_threads(4)
for moe in [False,True]:
 torch.manual_seed(42);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 model=MiniMindForCausalLM(MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=moe)).cuda().train()
 opt=torch.optim.AdamW(model.parameters(),lr=3e-4);x=torch.randint(3,6400,(64,340),device='cuda');started=time.time()
 for _ in range(2):
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):res=model(x,labels=x);loss=res.loss+res.aux_loss
  assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
  if moe:
   gates=[p.grad for n,p in model.named_parameters() if n.endswith('mlp.gate.weight')];assert len(gates)==8 and all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in gates)
  opt.step()
 torch.cuda.synchronize();rows.append(dict(model='moe' if moe else 'dense',batch=64,length=340,bf16=True,finite_loss=float(loss),finite_grad_norm=float(norm),peak_bytes=torch.cuda.max_memory_allocated(),seconds_per_microstep=(time.time()-started)/2))
 del opt,model,x,res,loss;gc.collect();torch.cuda.empty_cache()
write(ROOT/'smoke.json',dict(passed=True,models=rows,weights_discarded=True));print(json.dumps(rows),flush=True)
