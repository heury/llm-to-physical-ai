"""Operational runner using the repository's unmodified train_distillation.train_epoch."""
import hashlib,json,math,os,re,shutil,sys,tarfile,time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader,TensorDataset
from transformers import AutoTokenizer

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
import trainer.train_distillation as native
import dataset.lm_dataset as data
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from trainer.trainer_utils import setup_seed,SkipBatchSampler,lm_checkpoint
ROOT=Path(os.environ.get('DISTILL_ROOT','/shared-volume/minimind-distill-ultrachat-20261020'))
BASE=Path('/shared-volume/minimind-sft-ultrachat-200k-20261020')
RUNTIME=Path('/tmp/minimind-distill-20261020')
def save(name,value):
    tmp=ROOT/(name+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(ROOT/name)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def phase(value):save('phase.json',dict(phase=value,time=time.time()))
def finite(state):return all(torch.isfinite(v).all().item() for v in state.values() if torch.is_tensor(v))
def main():
    ROOT.mkdir(exist_ok=True);RUNTIME.mkdir(exist_ok=True)
    for n in ['out','checkpoints','tokenizer','source']:(ROOT/n).mkdir(exist_ok=True)
    phase('preparation');torch.set_num_threads(4);setup_seed(42)
    assert torch.cuda.is_available()
    gpu=torch.cuda.get_device_properties(0)
    assert 'B300' in gpu.name and 30*1024**3<gpu.total_memory<40*1024**3,(gpu.name,gpu.total_memory)
    torch.cuda.set_per_process_memory_fraction(.94)
    weights=BASE/'out/full_sft_768.pth'
    assert sha(weights)==json.loads((BASE/'evaluation_after.json').read_text())['checkpoint_sha256']
    manifest=json.loads((BASE/'dataset_manifest.json').read_text())
    for split in ['train_sft','test_sft']:
        source=BASE/'data'/(split+'.jsonl');assert sha(source)==manifest['splits'][split]['sha256']
    for n in ['train_sft.jsonl','validation.jsonl']:
        shutil.copyfile(BASE/'data'/n,RUNTIME/n)
    for p in (BASE/'tokenizer').iterdir():
        if p.is_file():shutil.copyfile(p,ROOT/'tokenizer'/p.name)
    tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True)
    # Source conversion already inserted a fixed system prompt; prevent random augmentation.
    data.pre_processing_chat=lambda conversations:conversations
    data.post_processing_chat=lambda text:text
    train=data.SFTDataset(str(RUNTIME/'train_sft.jsonl'),tok,max_length=1024)
    val=data.SFTDataset(str(RUNTIME/'validation.jsonl'),tok,max_length=1024)
    pairs=[val[i] for i in range(len(val))]
    assert len(train)==manifest['splits']['train_sft']['retained'] and len(pairs)==256
    assert all((y[1:]!=-100).sum()>=32 for x,y in pairs)
    vloader=DataLoader(TensorDataset(torch.stack([x for x,y in pairs]),torch.stack([y for x,y in pairs])),batch_size=8)
    tcfg=MiniMindConfig(hidden_size=768,num_hidden_layers=8)
    scfg=MiniMindConfig(hidden_size=768,num_hidden_layers=4)
    state=torch.load(weights,map_location='cpu',weights_only=True);assert finite(state)
    teacher=MiniMindForCausalLM(tcfg);teacher.load_state_dict(state,strict=True)
    student=MiniMindForCausalLM(scfg)
    mapped={}
    for key in student.state_dict():
        m=re.match(r'(model.layers.)(\d+)(\..*)',key)
        source=m[1]+str(int(m[2])*2)+m[3] if m else key
        mapped[key]=state[source]
    student.load_state_dict(mapped,strict=True);initial={k:v.clone() for k,v in student.state_dict().items()}
    del mapped,state
    teacher=teacher.cuda().bfloat16().eval().requires_grad_(False)
    student=student.cuda()
    teacher_hash=[(n,p.detach().float().sum().item()) for n,p in teacher.named_parameters()]
    tcfg.to_json_file(str(ROOT/'teacher_config.json'));scfg.to_json_file(str(ROOT/'config.json'))
    settings=dict(epochs=1,batch_size=32,accumulation_steps=2,learning_rate=5e-6,dtype='bfloat16',max_seq_len=1024,alpha=.5,temperature=1.5,grad_clip=1.,num_workers=4,log_interval=10,save_interval=250,student_hidden_size=768,student_num_layers=4,teacher_hidden_size=768,teacher_num_layers=8,student_use_moe=0,teacher_use_moe=0,seed=42,teacher_sha256=sha(weights),teacher_source=str(weights),layer_initialization=[0,2,4,6],torch=torch.__version__,cuda=torch.version.cuda,gpu=gpu.name,gpu_memory_bytes=gpu.total_memory)
    manifest.update(reused_verified_conversion=True,validation_documents=len(val),teacher_checkpoint_sha256=sha(weights))
    save('dataset_manifest.json',manifest)
    def evaluate(model):
        model.eval();nll=tokens=0
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for x,y in vloader:
                x=x.cuda();y=y.cuda();n=(y[:,1:]!=-100).sum().item()
                loss=model(x,labels=y).loss.float().item();assert math.isfinite(loss)
                nll+=loss*n;tokens+=n
        return dict(validation_loss=nll/tokens,assistant_tokens=tokens,validation_documents=len(val))
    phase('baseline_evaluation')
    before=evaluate(student);before['teacher']=evaluate(teacher);save('evaluation_before.json',before)
    # Actual longest-length forward/backward using native CE+KL; disposable gradients, no optimizer update.
    phase('memory_smoke');student.train();torch.cuda.reset_peak_memory_stats()
    batch=[train[i] for i in range(32)];x=torch.stack([x for x,y in batch]).cuda();y=torch.stack([y for x,y in batch]).cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=student(x).logits[:,:-1].contiguous()
        with torch.no_grad():target=teacher(x).logits[:,:-1].contiguous()
        labels=y[:,1:].contiguous();mask=labels.reshape(-1)!=-100
        ce=torch.nn.functional.cross_entropy(logits.reshape(-1,6400),labels.reshape(-1),ignore_index=-100)
        kl=native.distillation_loss(logits.reshape(-1,6400)[mask],target.reshape(-1,6400)[mask],temperature=1.5)
        loss=.5*ce+.5*kl
    assert mask.sum()>0 and torch.isfinite(loss)
    loss.backward();grad=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
    smoke=dict(loss=loss.item(),ce=ce.item(),kl=kl.item(),grad_norm=grad.item(),peak_cuda_bytes=torch.cuda.max_memory_allocated(),batch_size=32,optimizer_updates=0)
    assert smoke['peak_cuda_bytes']<gpu.total_memory*.90
    save('smoke.json',smoke);student.zero_grad(set_to_none=True)
    del batch,x,y,logits,target,labels,mask,ce,kl,loss;torch.cuda.empty_cache()
    args=SimpleNamespace(**{k:settings[k] for k in ['epochs','batch_size','accumulation_steps','learning_rate','dtype','max_seq_len','grad_clip','log_interval','save_interval','num_workers']},device='cuda:0',save_dir=str(ROOT/'out'),save_weight='full_dist')
    native.args=args;native.model=student;native.optimizer=torch.optim.AdamW(student.parameters(),lr=args.learning_rate)
    native.scaler=torch.amp.GradScaler('cuda',enabled=False)
    native.autocast_ctx=torch.autocast('cuda',dtype=torch.bfloat16)
    steps=math.ceil(len(train)/args.batch_size);settings['steps_per_epoch']=steps;settings['optimizer_updates']=math.ceil(steps/args.accumulation_steps)
    settings['student_parameters']=sum(p.numel() for p in student.parameters());settings['teacher_parameters']=sum(p.numel() for p in teacher.parameters())
    save('training_settings.json',settings)
    manifest['train_steps_per_epoch']=steps;save('dataset_manifest.json',manifest)
    updates=0
    def after_step(opt,argv,kwargs):
        nonlocal updates
        updates+=1
    native.optimizer.register_step_post_hook(after_step)
    def checkpoint(*a,**kw):
        kw['save_dir']=str(ROOT/'checkpoints');result=lm_checkpoint(*a,**kw)
        save('checkpoint.json',dict(step=kw.get('step'),optimizer_updates=updates,sha256=sha(ROOT/'out/full_dist_768.pth')))
        return result
    native.lm_checkpoint=checkpoint
    def logger(message):
        print(message,flush=True)
        m=re.search(r'\((\d+)/(\d+)\), loss: ([\d.e+-]+), ce: ([\d.e+-]+).*distill: ([\d.e+-]+).*epoch_time: ([\d.]+)',message)
        if m:
            row=dict(step=int(m[1]),total_steps=int(m[2]),loss=float(m[3]),ce=float(m[4]),kl=float(m[5]),eta_minutes=float(m[6]),optimizer_updates=updates,time=time.time())
            assert all(math.isfinite(row[n]) for n in ['loss','ce','kl'])
            save('progress.json',row)
            with (ROOT/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    native.Logger=logger
    setup_seed(42);indices=torch.randperm(len(train)).tolist()
    loader=DataLoader(train,batch_sampler=SkipBatchSampler(indices,args.batch_size,0),num_workers=args.num_workers,pin_memory=True)
    phase('training');student.train();start=time.time()
    native.train_epoch(0,loader,steps,teacher,scfg,0,None,settings['alpha'],settings['temperature'])
    assert updates==settings['optimizer_updates']
    phase('final_evaluation');after=evaluate(student)
    w=ROOT/'out/full_dist_768.pth';saved=torch.load(w,map_location='cpu',weights_only=True);assert finite(saved)
    reload=MiniMindForCausalLM(scfg);reload.load_state_dict(saved,strict=True)
    assert all(torch.equal(student.state_dict()[k].half().cpu(),v) for k,v in saved.items())
    changed=sum(not torch.equal(v,initial[k].half()) for k,v in saved.items());assert changed>0
    assert teacher_hash==[(n,p.detach().float().sum().item()) for n,p in teacher.named_parameters()]
    ck=torch.load(ROOT/'checkpoints/full_dist_768_resume.pth',map_location='cpu',weights_only=True)
    assert ck['epoch']==0 and ck['step']==steps and finite(ck['model'])
    assert all(torch.equal(v,ck['model'][k]) for k,v in saved.items())
    assert all(finite(v) for v in ck['optimizer']['state'].values())
    assert max(int(v['step']) for v in ck['optimizer']['state'].values())==updates
    after.update(steps=steps,optimizer_updates=updates,epochs=1,finite_weights=True,strict_load=True,teacher_frozen=True,changed_tensors=changed,checkpoint_sha256=sha(w),training_seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
    del ck,reload,saved,initial
    after['samples']=[];student.eval()
    for prompt in ['Explain photosynthesis in two sentences.','Write a Python function that adds two numbers.']:
        text=tok.apply_chat_template([{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,open_thinking=False)
        x=tok(text,return_tensors='pt').input_ids.cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            out=student.generate(inputs=x,max_new_tokens=128,do_sample=False,eos_token_id=tok.eos_token_id)
        after['samples'].append(dict(prompt=prompt,response=tok.decode(out[0,x.shape[1]:],skip_special_tokens=True)))
    save('evaluation_after.json',after)
    phase('packaging');shutil.copyfile(w,ROOT/'full_dist_768.pth')
    for name in ['trainer/train_distillation.py','trainer/trainer_utils.py','model/model_minimind.py','dataset/lm_dataset.py','scripts/runpod_distill/run.py']:
        dst=ROOT/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,dst)
    import subprocess
    (ROOT/'requirements-resolved.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    files=['full_dist_768.pth','config.json','teacher_config.json','dataset_manifest.json','training_settings.json','evaluation_before.json','evaluation_after.json','smoke.json','metrics.jsonl','requirements-resolved.txt']
    files += [str(p.relative_to(ROOT)) for d in ['tokenizer','source','checkpoints'] for p in (ROOT/d).rglob('*') if p.is_file()]
    save('SHA256.json',{n:sha(ROOT/n) for n in files})
    with tarfile.open(ROOT/'artifacts.tar.gz.partial','w:gz') as tar:
        for n in files+['SHA256.json']:tar.add(ROOT/n,arcname=n)
    (ROOT/'artifacts.tar.gz.partial').replace(ROOT/'artifacts.tar.gz')
    save('artifact.json',dict(sha256=sha(ROOT/'artifacts.tar.gz'),size=(ROOT/'artifacts.tar.gz').stat().st_size))
    phase('complete');print('DISTILL_COMPLETE',json.dumps(after),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'failure.txt').write_text(traceback.format_exc());raise
