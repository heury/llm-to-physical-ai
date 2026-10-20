import json,math,os,time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader,Subset
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from reward_adapter import BatchedRewardModel,GRPOPromptDataset
from trainer.rollout_engine import create_rollout_engine
import trainer.train_grpo as trainer
root=Path(os.environ['GRPO_ROOT']);runtime=Path(os.environ['GRPO_RUNTIME'])
torch.set_num_threads(4);torch.manual_seed(42)
batch_size=int(os.environ.get('GRPO_BATCH_SIZE','8'))
reward=BatchedRewardModel(runtime/'reward_model')
# Check the scorer on fixed preference pairs, without training the scorer.
with Path('/tmp/minimind-dpo-ultrafeedback-20261020/validation.jsonl').open(encoding='utf-8') as f:
    calibration=[json.loads(line) for i,line in enumerate(f) if i<128]
contexts=[];answers=[]
for row in calibration:
    for key in ['chosen','rejected']:
        contexts.append(row[key][:-1]);answers.append(row[key][-1]['content'])
scores=reward.score_batch(contexts,answers).view(-1,2)
agreement=float((scores[:,0]>scores[:,1]).float().mean());assert agreement>0.5,('Scorer preference sanity check failed',agreement)
assert scores.std()>1e-4
cfg=MiniMindConfig(hidden_size=768,num_hidden_layers=8,max_seq_len=1024)
base=Path('/shared-volume/minimind-dpo-ultrafeedback-20261020/out/dpo_768.pth')
state=torch.load(base,map_location='cpu',weights_only=True)
def load():
    model=MiniMindForCausalLM(cfg);model.load_state_dict(state,strict=True);return model.cuda()
model=load();ref=load().eval().requires_grad_(False)
tok=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True)
ds=GRPOPromptDataset(str(runtime/'train_prefs.jsonl'),tok,max_length=1024,thinking_ratio=0.)
indices=sorted(range(len(ds)),key=lambda i:ds.samples[i]['prompt_tokens'],reverse=True)[:batch_size]
loader=DataLoader(Subset(ds,indices),batch_size=batch_size)
smoke_dir=runtime/'smoke';(smoke_dir/'trainer').mkdir(parents=True,exist_ok=True);(smoke_dir/'out').mkdir(exist_ok=True)
os.chdir(smoke_dir/'trainer')
args=SimpleNamespace(device='cuda:0',num_generations=4,max_seq_len=768,max_gen_len=256,accumulation_steps=1,grad_clip=1.,log_interval=1,save_interval=1,save_dir=str(smoke_dir/'out'),save_weight='smoke',epochs=1,beta=.1,loss_type='grpo',epsilon=.2,epsilon_high=5.,debug_mode=False,debug_interval=20)
trainer.args=args;trainer.model=model;trainer.tokenizer=tok;trainer.lm_config=cfg;trainer.autocast_ctx=torch.autocast('cuda',dtype=torch.bfloat16)
trainer.optimizer=torch.optim.AdamW(model.parameters(),lr=3e-7)
trainer.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer,T_max=1,eta_min=3e-8)
engine=create_rollout_engine('torch',model,tok,'cuda:0',trainer.autocast_ctx)
class Recorder:
    id='smoke-only-not-an-online-run'
    def log(self,record):self.record=record
recorder=Recorder();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.time()
trainer.grpo_train_epoch(0,loader,1,engine,ref,reward,wandb=recorder)
torch.cuda.synchronize();metrics=recorder.record
(root/'smoke_metrics.json').write_text(json.dumps(metrics,indent=2))
assert metrics['grad_norm']>0 and math.isfinite(metrics['grad_norm'])
assert metrics['zero_variance_group_rate']<1
assert metrics['ratio_error_mean']<1e-3,metrics['ratio_error_mean']
assert metrics['kl_ref']<1e-4,metrics['kl_ref']
assert all(p.grad is None for p in ref.parameters())
checkpoint=torch.load(smoke_dir/'checkpoints/smoke_768_resume.pth',map_location='cpu',weights_only=True)
assert checkpoint['step']==1 and checkpoint['optimizer']['state']
changed=sum(not torch.equal(t.cpu(),state[k].float()) for k,t in model.state_dict().items());assert changed>0
manifest=json.loads((root/'dataset_manifest.json').read_text());manifest['steps_per_epoch']=math.ceil(manifest['splits']['train_prefs']['retained']/batch_size)
(root/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
settings=dict(epochs=1,batch_size=batch_size,num_generations=4,accumulation_steps=1,learning_rate=3e-7,beta=.1,epsilon=.2,loss_type='grpo',max_seq_len=768,max_gen_len=256,dtype='bfloat16',thinking_ratio=0.,temperature=1.,top_k=0,top_p=1.,save_interval=100,log_interval=1,reward_scale=5.,reward_clip=20.,repetition_penalty_cap=.5,hidden_size=768,num_hidden_layers=8)
(root/'training_settings.json').write_text(json.dumps(settings,indent=2))
report=dict(scorer_preference_pairs=128,scorer_preference_agreement=agreement,scorer_score_std=float(scores.std()),batch_size=batch_size,rollouts=batch_size*4,worst_prompt_tokens=max(ds.samples[i]['prompt_tokens'] for i in indices),peak_cuda_bytes=torch.cuda.max_memory_allocated(),elapsed_with_checkpoint_seconds=time.time()-start,changed_model_tensors=changed,production_base_unmodified=True,metrics=metrics,estimated_epoch_hours=metrics['step_seconds']*manifest['steps_per_epoch']/3600)
(root/'smoke.json').write_text(json.dumps(report,indent=2));print('SMOKE_SUCCESS',json.dumps(report),flush=True)
