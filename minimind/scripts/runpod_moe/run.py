"""FineWeb MoE pretraining using native trainer.train_pretrain.train_epoch."""
import gc,hashlib,json,math,os,re,shutil,subprocess,sys,tarfile,time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader,Subset,TensorDataset
from transformers import AutoTokenizer
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
import trainer.train_pretrain as native
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import setup_seed,SkipBatchSampler,lm_checkpoint
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
ROOT=Path('/shared-volume/minimind-moe-fineweb-20261020')
SOURCE=Path('/shared-volume/fineweb-edu-10BT-score-3.5')
RUNTIME=Path('/tmp/minimind-moe-20261020')
def save(n,v):
    p=ROOT/(n+'.tmp');p.write_text(json.dumps(v,indent=2,ensure_ascii=False));p.replace(ROOT/n)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def phase(n):save('phase.json',dict(phase=n,time=time.time()))
def finite(s):return all(torch.isfinite(v).all().item() for v in s.values() if torch.is_tensor(v))
def main():
    ROOT.mkdir(exist_ok=True);RUNTIME.mkdir(exist_ok=True)
    for n in ['out','checkpoints','tokenizer','source']:(ROOT/n).mkdir(exist_ok=True)
    phase('verify_source');torch.set_num_threads(4);setup_seed(42)
    gpu=torch.cuda.get_device_properties(0)
    assert 'B300' in gpu.name and 30*1024**3<gpu.total_memory<40*1024**3
    torch.cuda.set_per_process_memory_fraction(.94)
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    conversion=json.loads((SOURCE/'pretrain-score-3.5-conversion.json').read_text())
    assert manifest['status']=='complete' and manifest['kept']==conversion['documents']==1288228
    assert conversion['score_threshold']==3.5
    source=SOURCE/'pretrain-score-3.5.jsonl'
    # Verify the complete local copy, so both transfer and source integrity are checked.
    local=RUNTIME/'pretrain.jsonl';shutil.copyfile(source,local)
    assert sha(local)==conversion['sha256']
    for n,h in json.loads((SOURCE/'model/validation.json').read_text())['sha256'].items():assert sha(SOURCE/'model'/n)==h,n
    for p in (SOURCE/'model').iterdir():
        if p.is_file():shutil.copyfile(p,ROOT/'tokenizer'/p.name)
    tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True)
    assert len(tok)==6400 and (tok.pad_token_id,tok.bos_token_id,tok.eos_token_id)==(0,1,2)
    phase('prepare_dataset')
    ds=PretrainDataset(str(local),tok,max_length=1024);assert len(ds)==conversion['documents']
    # Fixed disjoint document indices; full remaining source is used, with no subset cap.
    g=torch.Generator().manual_seed(42);val_ids=torch.randperm(len(ds),generator=g)[:256].tolist();holdout=set(val_ids)
    train=Subset(ds,[i for i in range(len(ds)) if i not in holdout]);pairs=[ds[i] for i in val_ids]
    assert all((y[1:]!=-100).any() for x,y in pairs)
    vloader=DataLoader(TensorDataset(torch.stack([x for x,y in pairs]),torch.stack([y for x,y in pairs])),batch_size=8)
    save('dataset_manifest.json',dict(source=str(source),source_revision=manifest['revision'],score_threshold=3.5,source_sha256=conversion['sha256'],source_rows=len(ds),train_rows=len(train),validation_rows=256,validation_indices=val_ids,max_seq_len=1024,training_policy='One epoch over every non-heldout document; native 1022-token prefix plus BOS/EOS',train_validation_disjoint=True))
    cfg=MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=True)
    cfg.to_json_file(str(ROOT/'config.json'))
    # Benchmark disposable weights INCLUDING AdamW state. Production is freshly seeded afterwards.
    phase('memory_benchmark');bench=MiniMindForCausalLM(cfg).cuda().train();opt=torch.optim.AdamW(bench.parameters(),lr=3e-4)
    rows=[]
    for batch_size in [16,32]:
        opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        pairs_train=[train[i] for i in range(batch_size)];x=torch.stack([a for a,b in pairs_train]).cuda();y=torch.stack([b for a,b in pairs_train]).cuda()
        started=time.time()
        try:
            for repeat in range(2):
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    res=bench(x,labels=y);loss=res.loss+res.aux_loss
                assert torch.isfinite(loss) and res.aux_loss.item()>0
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(bench.parameters(),1.,error_if_nonfinite=True)
                routers=[p.grad for n,p in bench.named_parameters() if n.endswith('mlp.gate.weight')]
                assert len(routers)==8 and all(v is not None and torch.isfinite(v).all() and v.abs().sum()>0 for v in routers)
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in bench.named_parameters() if '.experts.' in n)
                opt.step()
            torch.cuda.synchronize()
            row=dict(batch_size=batch_size,peak_cuda_bytes=torch.cuda.max_memory_allocated(),loss=loss.item(),ce=res.loss.item(),aux=res.aux_loss.item(),grad_norm=norm.item(),seconds_per_batch=(time.time()-started)/2,routers_nonzero_finite=True,expert_gradients_finite=True)
            rows.append(row);print('BENCHMARK',json.dumps(row),flush=True)
            if row['peak_cuda_bytes']>gpu.total_memory*.85:break
        except torch.OutOfMemoryError:
            print('Benchmark OOM at',batch_size,flush=True);break
        finally:
            opt.zero_grad(set_to_none=True);del x,y;gc.collect();torch.cuda.empty_cache()
    safe=[r for r in rows if r['peak_cuda_bytes']<gpu.total_memory*.85];assert safe,rows
    chosen=max(safe,key=lambda r:r['batch_size']/r['seconds_per_batch']);batch=chosen['batch_size']
    del opt,bench,res,loss,routers,pairs_train;gc.collect();torch.cuda.empty_cache();setup_seed(42)
    model=MiniMindForCausalLM(cfg).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    steps=math.ceil(len(train)/batch);accum=128//batch
    settings=dict(epochs=1,batch_size=batch,accumulation_steps=accum,effective_batch_size=batch*accum,learning_rate=3e-4,warmup_microsteps=min(1000,int(steps*.02)),dtype='bfloat16',max_seq_len=1024,hidden_size=768,num_hidden_layers=8,use_moe=True,num_experts=4,num_experts_per_tok=1,router_aux_loss_coef=cfg.router_aux_loss_coef,grad_clip=1.,num_workers=4,log_interval=10,save_interval=250,from_weight='none',seed=42,steps_per_epoch=steps,optimizer_updates=math.ceil(steps/accum),model_parameters=sum(p.numel() for p in model.parameters()),active_parameters=sum(p.numel() for n,p in model.named_parameters() if '.experts.' not in n or '.experts.0.' in n),gpu=gpu.name,gpu_memory_bytes=gpu.total_memory,torch=torch.__version__,cuda=torch.version.cuda,benchmark=rows,estimated_training_hours=chosen['seconds_per_batch']*steps/3600)
    save('training_settings.json',settings);save('smoke.json',chosen)
    def evaluate(m):
        m.eval();nll=tokens=0
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for x,y in vloader:
                x=x.cuda();y=y.cuda();count=(y[:,1:]!=-100).sum().item();loss=m(x,labels=y).loss.float().item();assert math.isfinite(loss)
                nll+=loss*count;tokens+=count
        return dict(validation_loss=nll/tokens,validation_documents=256,tokens=tokens,perplexity=math.exp(nll/tokens))
    phase('baseline_evaluation');save('evaluation_before.json',evaluate(model));model.train()
    args=SimpleNamespace(**{k:settings[k] for k in ['epochs','batch_size','accumulation_steps','learning_rate','dtype','max_seq_len','grad_clip','log_interval','save_interval','num_workers']},device='cuda:0',save_dir=str(ROOT/'out'),save_weight='pretrain')
    native.args=args;native.model=model;native.optimizer=optimizer;native.lm_config=cfg
    native.scaler=torch.amp.GradScaler('cuda',enabled=False);native.autocast_ctx=torch.autocast('cuda',dtype=torch.bfloat16)
    updates=0
    def checkpoint(*a,**kw):
        kw['save_dir']=str(ROOT/'checkpoints');result=lm_checkpoint(*a,**kw)
        p=ROOT/'checkpoints/pretrain_768_moe.pth'
        save('checkpoint.json',dict(step=kw.get('step'),optimizer_updates=updates,sha256=sha(p)))
        return result
    native.lm_checkpoint=checkpoint
    def post_step(opt,argv,kwargs):
        nonlocal updates
        updates+=1
        if updates==1:
            checkpoint(cfg,weight='pretrain',model=model,optimizer=optimizer,scaler=native.scaler,epoch=0,step=accum)
            shutil.copyfile(ROOT/'checkpoints/pretrain_768_moe.pth',ROOT/'out/pretrain_768_moe.pth')
    optimizer.register_step_post_hook(post_step)
    def lr(step,total,peak):
        warm=settings['warmup_microsteps']
        if step<=warm:return peak*step/max(warm,1)
        return peak*(.1+.45*(1+math.cos(math.pi*(step-warm)/max(total-warm,1))))
    native.get_lr=lr
    def logger(message):
        print(message,flush=True)
        m=re.search(r'\((\d+)/(\d+)\), loss: ([\d.e+-]+), logits_loss: ([\d.e+-]+), aux_loss: ([\d.e+-]+).*epoch_time: ([\d.]+)',message)
        if m:
            row=dict(step=int(m[1]),total_steps=int(m[2]),loss=float(m[3]),ce=float(m[4]),aux=float(m[5]),eta_minutes=float(m[6]),optimizer_updates=updates,time=time.time());assert all(math.isfinite(row[n]) for n in ['loss','ce','aux'])
            save('progress.json',row)
            with (ROOT/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    native.Logger=logger
    setup_seed(42);indices=torch.randperm(len(train)).tolist()
    loader=DataLoader(train,batch_sampler=SkipBatchSampler(indices,batch,0),num_workers=4,pin_memory=True)
    phase('training');start=time.time();model.train();native.train_epoch(0,loader,steps)
    assert updates==settings['optimizer_updates']
    phase('final_evaluation');after=evaluate(model)
    p=ROOT/'out/pretrain_768_moe.pth';state=torch.load(p,map_location='cpu',weights_only=True);assert finite(state)
    fresh=MiniMindForCausalLM(cfg);fresh.load_state_dict(state,strict=True)
    assert all(torch.equal(v,model.state_dict()[k].half().cpu()) for k,v in state.items())
    resume=torch.load(ROOT/'checkpoints/pretrain_768_moe_resume.pth',map_location='cpu',weights_only=True)
    assert resume['step']==steps and resume['epoch']==0 and all(torch.equal(v,resume['model'][k]) for k,v in state.items())
    assert all(finite(v) for v in resume['optimizer']['state'].values())
    assert max(int(v['step']) for v in resume['optimizer']['state'].values())==updates
    after.update(steps=steps,epochs=1,optimizer_updates=updates,finite_weights=True,finite_optimizer=True,strict_load=True,checkpoint_sha256=sha(p),training_seconds=time.time()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
    del state,fresh,resume
    after['samples']=[]
    for prompt in ['The purpose of education is','Science is important because']:
        x=tok(prompt,return_tensors='pt').input_ids.cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model.generate(inputs=x,max_new_tokens=96,do_sample=False,eos_token_id=tok.eos_token_id)
        after['samples'].append(dict(prompt=prompt,completion=tok.decode(out[0,x.shape[1]:],skip_special_tokens=True)))
    save('evaluation_after.json',after);phase('packaging');shutil.copyfile(p,ROOT/'pretrain_768_moe.pth')
    for n in ['trainer/train_pretrain.py','trainer/trainer_utils.py','model/model_minimind.py','dataset/lm_dataset.py','scripts/runpod_moe/run.py']:
        dest=ROOT/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/n,dest)
    (ROOT/'requirements-resolved.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    files=['pretrain_768_moe.pth','config.json','dataset_manifest.json','training_settings.json','evaluation_before.json','evaluation_after.json','smoke.json','metrics.jsonl','requirements-resolved.txt']
    files += [str(p.relative_to(ROOT)) for d in ['tokenizer','source','checkpoints'] for p in (ROOT/d).rglob('*') if p.is_file()]
    save('SHA256.json',{n:sha(ROOT/n) for n in files})
    with tarfile.open(ROOT/'artifacts.tar.gz.partial','w:gz') as tar:
        for n in files+['SHA256.json']:tar.add(ROOT/n,arcname=n)
    (ROOT/'artifacts.tar.gz.partial').replace(ROOT/'artifacts.tar.gz');save('artifact.json',dict(sha256=sha(ROOT/'artifacts.tar.gz'),size=(ROOT/'artifacts.tar.gz').stat().st_size))
    phase('complete');print('MOE_COMPLETE',json.dumps(after),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'failure.txt').write_text(traceback.format_exc());raise
