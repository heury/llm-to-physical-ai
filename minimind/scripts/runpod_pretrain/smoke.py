"""Credential-free CUDA forward/backward verification, no production training."""
import gc,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
root=Path('/shared-volume/minimind-pretrain-rtx5090-20261023')
source=Path('/shared-volume/fineweb-edu-10BT-score-3.5')
torch.set_num_threads(4)
tokenizer=AutoTokenizer.from_pretrained(source/'model',local_files_only=True)
with (source/'pretrain-score-3.5.jsonl').open() as f:row=json.loads(next(f))
tokens=tokenizer(row['text'],add_special_tokens=False,truncation=True,max_length=338).input_ids
ids=[tokenizer.bos_token_id]+tokens+[tokenizer.eos_token_id]
x=torch.tensor([ids+[tokenizer.pad_token_id]*(340-len(ids))]*32,device='cuda')
y=x.clone();y[y==tokenizer.pad_token_id]=-100
reports=[]
for moe in [False,True]:
 torch.manual_seed(42)
 m=MiniMindForCausalLM(MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=moe)).cuda()
 opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
 torch.cuda.reset_peak_memory_stats()
 with torch.autocast('cuda',dtype=torch.bfloat16):
  result=m(x,labels=y);loss=result.loss+result.aux_loss
 assert torch.isfinite(loss)
 loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
 opt.step();torch.cuda.synchronize()
 report=dict(model='moe' if moe else 'dense',batch_size=32,max_seq_len=340,loss=float(loss),grad_norm=float(norm),peak_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),forward_backward_optimizer=True)
 reports.append(report);print(json.dumps(report),flush=True)
 del m,opt,result,loss;gc.collect();torch.cuda.empty_cache()
(root/'smoke.json').write_text(json.dumps(reports,indent=2))
