"""Full-parameter DPO with a frozen base reference and held-out selection."""
import argparse,contextlib,hashlib,json,math,os,random,shutil,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

SETTINGS={'epochs':2,'batch_pairs':32,'learning_rate':5e-7,'beta':0.1,'warmup_ratio':0.1,'weight_decay':0.0,'grad_clip':1.0,'precision':'bfloat16 autocast / float32 master weights','sequence_storage':340,'max_nonpadding_tokens':339,'seed':42}
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def statehash(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def logps(model,arr,device):
 a=torch.as_tensor(np.asarray(arr).copy(),device=device,dtype=torch.long);b=len(a)
 x=a[:,:,0,:-1].reshape(b*2,-1);y=a[:,:,1,1:].reshape(b*2,-1);mask=y!=-100
 ctx=torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') else contextlib.nullcontext()
 with ctx:logits=model(x,use_cache=False).logits
 lp=F.log_softmax(logits.float(),dim=-1).gather(-1,y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
 return (lp*mask).sum(-1).reshape(b,2),mask.sum(-1).reshape(b,2)
def metrics(scores,ref,counts,beta):
 s=torch.tensor(scores);r=torch.tensor(ref);n=torch.tensor(counts);margin=beta*((s[:,0]-s[:,1])-(r[:,0]-r[:,1]));cnll=float(-s[:,0].sum()/n[:,0].sum());rnll=float(-s[:,1].sum()/n[:,1].sum())
 return {'pairs':len(s),'dpo_loss':float(-F.logsigmoid(margin).mean()),'reference_relative_margin':float(margin.mean()),'reference_relative_accuracy':float((margin>1e-6).float().mean()),'reference_relative_tie_rate':float((margin.abs()<=1e-6).float().mean()),'raw_sequence_preference_accuracy':float((s[:,0]>s[:,1]).float().mean()),'per_token_preference_accuracy':float((s[:,0]/n[:,0]>s[:,1]/n[:,1]).float().mean()),'chosen_nll':cnll,'chosen_ppl':math.exp(cnll),'rejected_nll':rnll,'rejected_ppl':math.exp(rnll)}
@torch.inference_mode()
def score(model,arr,device,batch=32):
 model.eval();scores=[];counts=[]
 for i in range(0,len(arr),batch):
  a,n=logps(model,arr[i:i+batch],device);scores.append(a.cpu().numpy());counts.append(n.cpu().numpy())
 return np.concatenate(scores),np.concatenate(counts)
def chat(tok,msgs,generation=False):return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=generation,open_thinking=False).replace('<think>\n\n</think>\n\n','')
@torch.inference_mode()
def generate(model,tok,questions,device):
 model.eval();result=[]
 for item in questions:
  text=chat(tok,item['messages'],True);ids=tok(text,add_special_tokens=False,return_tensors='pt').input_ids.to(device);limit=min(128,340-ids.shape[1]);assert limit>0
  with torch.autocast('cuda',dtype=torch.bfloat16):
   out=model.generate(ids,max_new_tokens=limit,do_sample=False,temperature=1.0,top_p=1.0,top_k=0,repetition_penalty=1.0,eos_token_id=tok.eos_token_id)
  result.append(dict(item,output=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True),generated_tokens=int(out.shape[1]-ids.shape[1]),hit_limit=out.shape[1]-ids.shape[1]==limit))
 return result

def main(root):
 root=Path(root);work=root/'results';work.mkdir(exist_ok=True);device='cuda:0';assert torch.cuda.is_available()
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(4)
 write(work/'settings.json',SETTINGS);write(work/'environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'transformers':__import__('transformers').__version__})
 def status(phase,**kw):write(work/'status.json',dict(phase=phase,timestamp=time.time(),**kw));print(phase,kw,flush=True)
 status('load_model')
 tok=AutoTokenizer.from_pretrained(root/'base',local_files_only=True)
 policy=AutoModelForCausalLM.from_pretrained(root/'base',trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32).to(device)
 ref=AutoModelForCausalLM.from_pretrained(root/'base',trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
 assert sum(p.numel() for p in policy.parameters())==63912192
 assert all(p.data_ptr()!=q.data_ptr() for p,q in zip(policy.parameters(),ref.parameters()))
 ref_hash=statehash(ref);arrays={s:np.load(root/'data'/f'{s}.npy',mmap_mode='r') for s in ['train','validation','test']};references={};target_counts={}
 for split,a in arrays.items():
  status('cache_reference',split=split);references[split],target_counts[split]=score(ref,a,device);np.save(work/f'reference-{split}.npy',references[split])
 before={s:metrics(references[s],references[s],target_counts[s],SETTINGS['beta']) for s in arrays};write(work/'before.json',before)
 # A real-batch backward check without changing weights.
 policy.train();p,n=logps(policy,arrays['train'][:32],device);r=torch.tensor(references['train'][:32],device=device);margin=SETTINGS['beta']*((p[:,0]-p[:,1])-(r[:,0]-r[:,1]));loss=-F.logsigmoid(margin).mean()
 assert abs(loss.item()-math.log(2))<1e-4;loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0);assert torch.isfinite(norm) and norm>0
 assert all(p.grad is None for p in ref.parameters());policy.zero_grad(set_to_none=True);write(work/'smoke.json',{'identical_policy_reference_loss':loss.item(),'expected':math.log(2),'gradient_norm':float(norm),'reference_frozen':True});del p,n,r,margin,loss
 optimizer=torch.optim.AdamW(policy.parameters(),lr=SETTINGS['learning_rate'],betas=(.9,.95),weight_decay=0.0)
 steps_epoch=math.ceil(len(arrays['train'])/32);total=steps_epoch*2;step=0;best=math.inf;best_epoch=None;started=time.time();history=[]
 with (work/'metrics.jsonl').open('w',buffering=1) as log:
  for epoch in range(1,3):
   policy.train();order=np.random.default_rng(42+epoch).permutation(len(arrays['train']));status('training',epoch=epoch,steps=total)
   for start in range(0,len(order),32):
    idx=order[start:start+32];step+=1;warmup=max(1,math.ceil(total*.1));scale=step/warmup if step<=warmup else .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))
    lr=SETTINGS['learning_rate']*scale
    for group in optimizer.param_groups:group['lr']=lr
    optimizer.zero_grad(set_to_none=True);p,n=logps(policy,arrays['train'][idx],device);r=torch.tensor(references['train'][idx],device=device)
    margin=SETTINGS['beta']*((p[:,0]-p[:,1])-(r[:,0]-r[:,1]));loss=-F.logsigmoid(margin).mean();assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0);assert torch.isfinite(norm);optimizer.step()
    record={'step':step,'epoch':epoch,'loss':loss.item(),'margin':margin.mean().item(),'reward_accuracy':(margin>0).float().mean().item(),'learning_rate':lr,'gradient_norm':float(norm),'elapsed_seconds':time.time()-started};log.write(json.dumps(record)+'\n')
    if step%10==0:status('training',**record)
   s,n=score(policy,arrays['validation'],device);val=metrics(s,references['validation'],n,SETTINGS['beta']);history.append({'epoch':epoch,'validation':val});write(work/'validation_history.json',history)
   if val['dpo_loss']<best:
    best=val['dpo_loss'];best_epoch=epoch;torch.save({k:v.detach().cpu() for k,v in policy.state_dict().items()},work/'best.pth')
  train_seconds=time.time()-started
 status('evaluate_selected',epoch=best_epoch);policy.load_state_dict(torch.load(work/'best.pth',map_location=device,weights_only=True),strict=True)
 after={}
 for split in arrays:
  s,n=score(policy,arrays[split],device);after[split]=metrics(s,references[split],n,SETTINGS['beta']);np.save(work/f'policy-{split}.npy',s)
 write(work/'after.json',after);assert statehash(ref)==ref_hash
 questions=[{'name':f'general_{i}','messages':[{'role':'user','content':q}]} for i,q in enumerate(['Why is the sky blue?','What is 17 + 25?','Rewrite politely: Send me the report now.','Give two ways to save water at home.','What is the capital of France?','Explain photosynthesis in one sentence.','Translate "Good morning" into Spanish.','List two even numbers.'])]
 tests=[json.loads(x) for x in (root/'data/test.jsonl').read_text().splitlines()]
 questions.extend({'name':f'heldout_{i}','source_row':r['source_row'],'messages':r['chosen'][:-1]} for i,r in enumerate(tests[:8]))
 status('sample_generation');samples_before=generate(ref,tok,questions,device);samples_after=generate(policy,tok,questions,device)
 write(work/'samples.json',[{'name':b['name'],'prompt':b['messages'],'before':b['output'],'after':a['output'],'before_tokens':b['generated_tokens'],'after_tokens':a['generated_tokens'],'before_hit_limit':b['hit_limit'],'after_hit_limit':a['hit_limit']} for b,a in zip(samples_before,samples_after)])
 changed=sum(int((p!=q).sum()) for p,q in zip(policy.parameters(),ref.parameters()));delta=math.sqrt(sum(float((p.detach()-q).square().sum()) for p,q in zip(policy.parameters(),ref.parameters())))
 assert changed>0 and delta>0
 out=Path('/tmp/dpo-model-export');policy.config.model_version='v1.0';policy.config.finetuning_task='dpo-ultrafeedback-short';policy.config.base_model_name_or_path='caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0'
 policy.__class__.register_for_auto_class('AutoModelForCausalLM');policy.config.__class__.register_for_auto_class();policy.save_pretrained(out,safe_serialization=True);tok.save_pretrained(out)
 restored=AutoModelForCausalLM.from_pretrained(out,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32).to(device).eval()
 assert all(torch.equal(v,restored.state_dict()[k]) for k,v in policy.state_dict().items());a,_=score(restored,arrays['test'][:32],device);b,_=score(policy,arrays['test'][:32],device);np.testing.assert_array_equal(a,b)
 with (out/'model.safetensors').open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 for p in out.rglob('*'):
  if p.is_file():
   dst=root/'model'/p.relative_to(out);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
 summary={'complete':True,'training_epochs':2,'selected_epoch':best_epoch,'optimizer_steps':step,'training_seconds':train_seconds,'before':before,'after':after,'reference_unchanged':True,'changed_parameter_elements':changed,'parameter_delta_l2':delta,'model_sha256':digest,'export_dtype':'float32','reload_weights_and_scores_exact':True,'peak_gpu_memory_gb':torch.cuda.max_memory_allocated()/1e9,'completed_at':time.time()}
 write(work/'completion.json',summary);status('complete',**summary)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
 try:main(a.root)
 except BaseException as e:
  write(Path(a.root)/'results/failed.json',{'error':repr(e),'timestamp':time.time()});raise
