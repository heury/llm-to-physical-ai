"""Sequential Dense/MoE pretraining with the native MiniMind training loop."""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path('/shared-volume/minimind-pretrain-rtx5090-20261023')
RUNTIME = Path('/tmp/minimind-pretrain-rtx5090')
SOURCE = Path('/shared-volume/fineweb-edu-10BT-score-3.5')
REPO = RUNTIME / 'repo'
sys.path.insert(0, str(REPO))
os.environ.update(TOKENIZERS_PARALLELISM='false', HF_HOME=str(RUNTIME/'hf'),
                  HF_DATASETS_CACHE=str(RUNTIME/'datasets'), WANDB_MODE='online',
                  WANDB_DIR=str(ROOT/'wandb'), WANDB_CACHE_DIR=str(RUNTIME/'wandb-cache'),
                  OMP_NUM_THREADS='4')

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)

def phase(name, **extra):
    write(ROOT/'status.json', dict(phase=name, timestamp=time.time(), **extra))
    print('PHASE', name, flush=True)

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    import torch
    import wandb
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, Subset
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.trainer_utils import setup_seed, SkipBatchSampler, lm_checkpoint
    import trainer.train_pretrain as native

    ROOT.mkdir(exist_ok=True)
    (ROOT/'wandb').mkdir(exist_ok=True)
    assert not (ROOT/'completion.json').exists(), 'Run already complete'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    gpu = torch.cuda.get_device_properties(0)
    assert '5090' in gpu.name and torch.cuda.is_bf16_supported()
    torch.cuda.set_per_process_memory_fraction(.94)
    os.environ['WANDB_API_KEY'] = (RUNTIME/'wandb.key').read_text().strip()
    (RUNTIME/'wandb.key').unlink()
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    conversion = json.loads((SOURCE/'pretrain-score-3.5-conversion.json').read_text())
    assert manifest['status'] == 'complete' and manifest['threshold'] == 3.5
    assert all(s['min_score'] >= 3.5 for s in manifest['shards'])
    phase('copy_and_verify_data')
    local = RUNTIME/'pretrain.jsonl'
    if not local.exists():
        shutil.copyfile(SOURCE/'pretrain-score-3.5.jsonl', local)
    assert sha(local) == conversion['sha256'], 'Dataset checksum mismatch'
    tok_dir = ROOT/'tokenizer'
    tok_dir.mkdir(exist_ok=True)
    token_manifest = json.loads((SOURCE/'model/dataset_manifest.json').read_text())
    for name, digest in token_manifest.get('tokenizer_sha256', {}).items():
        assert sha(SOURCE/'model'/name) == digest, name
    for p in (SOURCE/'model').iterdir():
        if p.is_file(): shutil.copyfile(p, tok_dir/p.name)
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    assert len(tokenizer) == 6400
    assert (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id) == (0,1,2)
    phase('load_dataset')
    ds = PretrainDataset(str(local), tokenizer, max_length=340)
    assert len(ds) == conversion['documents'] == manifest['kept']
    ids = torch.randperm(len(ds), generator=torch.Generator().manual_seed(42))[:256].tolist()
    heldout = set(ids)
    train = Subset(ds, [i for i in range(len(ds)) if i not in heldout])
    validation = Subset(ds, ids)
    write(ROOT/'dataset_manifest.json', dict(source=str(SOURCE), sha256=conversion['sha256'],
          source_revision=manifest['revision'], score_threshold=3.5, source_documents=len(ds),
          training_documents=len(train), validation_documents=len(ids), validation_indices=ids,
          max_seq_len=340, policy='Each document: first 338 tokens plus BOS/EOS; two epochs over every training document'))
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=(ROOT/'requirements-resolved.txt').open('w'), check=True)
    completions = []

    for moe in [False, True]:
        name = 'moe' if moe else 'dense'
        out = ROOT/name
        for folder in ['out','checkpoints']: (out/folder).mkdir(parents=True, exist_ok=True)
        phase(name+'_benchmark')
        cfg = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=moe)
        cfg.to_json_file(str(out/'config.json'))
        previous_settings = json.loads((out/'training_settings.json').read_text()) if (out/'training_settings.json').exists() else None
        if previous_settings:
            candidates = previous_settings['benchmark']
            batch = previous_settings['batch_size']
            accum = previous_settings['accumulation_steps']
            chosen = next(r for r in candidates if r['batch_size'] == batch)
        else:
            setup_seed(42)
            bench = MiniMindForCausalLM(cfg).cuda().train()
            opt = torch.optim.AdamW(bench.parameters(), lr=3e-4)
            candidates = []
            for batch in [32,64,128]:
                x = y = res = loss = None
                opt.zero_grad(set_to_none=True)
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                try:
                    pairs = [train[i] for i in range(batch)]
                    x = torch.stack([a for a,b in pairs]).cuda()
                    y = torch.stack([b for a,b in pairs]).cuda()
                    started = time.time()
                    for repeat in range(2):
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            res = bench(x, labels=y)
                            loss = res.loss + res.aux_loss
                        assert torch.isfinite(loss)
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(bench.parameters(), 1., error_if_nonfinite=True)
                        if moe:
                            routers = [p.grad for n,p in bench.named_parameters() if n.endswith('mlp.gate.weight')]
                            assert len(routers) == 8 and all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in routers)
                            del routers
                        opt.step()
                    torch.cuda.synchronize()
                    row = dict(batch_size=batch, peak_bytes=torch.cuda.max_memory_allocated(),
                               seconds_per_batch=(time.time()-started)/2, loss=float(loss), grad_norm=float(norm))
                    candidates.append(row)
                    print('BENCHMARK', name, json.dumps(row), flush=True)
                    if row['peak_bytes'] > gpu.total_memory*.85: break
                except torch.OutOfMemoryError:
                    print('BENCHMARK OOM', name, batch, flush=True)
                    break
                finally:
                    opt.zero_grad(set_to_none=True)
                    del x,y,res,loss
                    gc.collect(); torch.cuda.empty_cache()
            safe = [r for r in candidates if r['peak_bytes'] < gpu.total_memory*.85]
            assert safe, candidates
            chosen = max(safe, key=lambda r:r['batch_size']/r['seconds_per_batch'])
            batch = chosen['batch_size']; accum = 256//batch
            del bench,opt
        gc.collect();torch.cuda.empty_cache();setup_seed(42)
        model = MiniMindForCausalLM(cfg).cuda()
        optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
        steps = math.ceil(len(train)/batch)
        settings = dict(epochs=2, batch_size=batch, accumulation_steps=accum, effective_batch_size=256,
                        max_seq_len=340, learning_rate=3e-4, warmup_microsteps=min(1000,max(1,int(steps*.02))),
                        dtype='bfloat16', hidden_size=768,num_hidden_layers=8,use_moe=moe,
                        grad_clip=1., num_workers=4,log_interval=10,save_interval=250,
                        seed=42,from_weight='none',steps_per_epoch=steps,optimizer_updates=2*math.ceil(steps/accum),
                        gpu=gpu.name,gpu_memory_bytes=gpu.total_memory,model_parameters=sum(p.numel() for p in model.parameters()),
                        torch=torch.__version__,cuda=torch.version.cuda,benchmark=candidates,
                        estimated_training_hours=2*chosen['seconds_per_batch']*steps/3600,
                        num_experts=cfg.num_experts if moe else None,num_experts_per_tok=cfg.num_experts_per_tok if moe else None)
        write(out/'training_settings.json', settings)
        args = SimpleNamespace(**{k:settings[k] for k in ['epochs','batch_size','accumulation_steps','learning_rate','dtype','max_seq_len','grad_clip','num_workers','log_interval','save_interval']}, device='cuda:0',save_dir=str(out/'out'),save_weight='pretrain')
        native.args=args;native.model=model;native.optimizer=optimizer;native.lm_config=cfg
        native.scaler=torch.amp.GradScaler('cuda',enabled=False)
        native.autocast_ctx=torch.autocast('cuda',dtype=torch.bfloat16)
        suffix='_moe' if moe else ''
        weights=out/f'out/pretrain_768{suffix}.pth'
        updates=0
        start_epoch = start_step = 0
        resume_path = out/f'checkpoints/pretrain_768{suffix}_resume.pth'
        if resume_path.exists():
            saved = torch.load(resume_path, map_location='cpu', weights_only=False)
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            native.scaler.load_state_dict(saved.get('scaler', {}))
            updates = int(max(v['step'].item() for v in optimizer.state.values() if 'step' in v))
            start_epoch = saved['epoch']
            # Checkpoints may be taken between optimizer updates. Replay only
            # the uncommitted accumulation window: its gradients were not saved.
            start_step = saved['step']
            if start_step == steps:
                start_epoch += 1
                start_step = 0
            else:
                start_step -= start_step % accum
            write(out/'resume.json', dict(checkpoint_step=saved['step'], restart_epoch=start_epoch,
                  restart_step=start_step, optimizer_updates=updates, wandb_id=saved['wandb_id'],
                  epochs=2, learning_rate_policy='Extend remaining cosine schedule to total two epochs'))
            del saved
        def evaluate():
            model.eval();nll=tokens=0
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for x,y in DataLoader(validation,batch_size=16,num_workers=2):
                    x=x.cuda();y=y.cuda();n=int((y[:,1:]!=-100).sum())
                    l=float(model(x,labels=y).loss);assert math.isfinite(l)
                    nll+=l*n;tokens+=n
            model.train()
            return dict(loss=nll/tokens,perplexity=math.exp(nll/tokens),tokens=tokens)
        def lr(step,total,peak):
            warm=settings['warmup_microsteps']
            return peak*step/warm if step<=warm else peak*(.1+.45*(1+math.cos(math.pi*(step-warm)/max(total-warm,1))))
        native.get_lr=lr
        def logger(message):
            print(name, message,flush=True)
            match=re.search(r'\((\d+)/(\d+)\), loss: ([\d.e+-]+).*epoch_time: ([\d.]+)',message)
            if match:
                row=dict(model=name,epoch=current_epoch+1,epochs=args.epochs,global_step=current_epoch*steps+int(match[1]),step=int(match[1]),total_steps=int(match[2]),loss=float(match[3]),eta_minutes=float(match[4]),optimizer_updates=updates,timestamp=time.time())
                assert math.isfinite(row['loss'])
                write(out/'progress.json',row)
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        native.Logger=logger
        prior_run = json.loads((out/'wandb_run.json').read_text()) if (out/'wandb_run.json').exists() else None
        run = wandb.init(project='MiniMind-FineWeb-Pretrain', name=f'RTX5090-{name}-768x8-seq340-20261023',
                         id=prior_run['id'] if prior_run else None, resume='must' if prior_run else None,
                         dir=str(ROOT/'wandb'))
        run.config.update(settings, allow_val_change=True)
        write(out/'wandb_run.json',dict(id=run.id,url=run.url,project=run.project))
        def checkpoint(*a,**kw):
            kw['save_dir']=str(out/'checkpoints')
            lm_checkpoint(*a,**kw)
            p=out/f'checkpoints/pretrain_768{suffix}.pth'
            write(out/'checkpoint.json',dict(epoch=kw.get('epoch',0)+1,epochs=args.epochs,step=kw.get('step'),optimizer_updates=updates,sha256=sha(p),wandb_id=run.id))
        native.lm_checkpoint=checkpoint
        def post_step(opt,argv,kwargs):
            nonlocal updates
            updates+=1
            if updates==1:
                checkpoint(cfg,weight='pretrain',model=model,optimizer=optimizer,scaler=native.scaler,epoch=0,step=accum,wandb=run)
                shutil.copyfile(out/f'checkpoints/pretrain_768{suffix}.pth',weights)
        optimizer.register_step_post_hook(post_step)
        if (out/'evaluation_before.json').exists():
            before=json.loads((out/'evaluation_before.json').read_text())
        else:
            before=evaluate();write(out/'evaluation_before.json',before)
            run.log({'validation/before_loss':before['loss']})
        phase(name+'_training',wandb_url=run.url,settings=settings)
        started=time.time()
        for current_epoch in range(start_epoch,args.epochs):
            setup_seed(42+current_epoch)
            indices=torch.randperm(len(train)).tolist()
            skip = start_step if current_epoch == start_epoch else 0
            loader=DataLoader(train,batch_sampler=SkipBatchSampler(indices,batch,skip),num_workers=4,pin_memory=True)
            optimizer.zero_grad(set_to_none=True)
            native.train_epoch(current_epoch,loader,steps,start_step=skip,wandb=run)
            # Keep each completed epoch for comparison and rollback.
            shutil.copyfile(weights,out/f'out/pretrain_768{suffix}_epoch{current_epoch+1}.pth')
            write(out/f'epoch{current_epoch+1}.json',dict(epoch=current_epoch+1,steps=steps,optimizer_updates=updates))
        assert updates == settings['optimizer_updates']
        phase(name+'_verification')
        after=evaluate();write(out/'evaluation_after.json',after)
        state=torch.load(weights,map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in state.values())
        fresh=MiniMindForCausalLM(cfg);fresh.load_state_dict(state,strict=True)
        resume_path=out/f'checkpoints/pretrain_768{suffix}_resume.pth'
        resume=torch.load(resume_path,map_location='cpu',weights_only=False)
        assert resume['epoch']==args.epochs-1 and resume['step']==steps and resume['wandb_id']==run.id
        assert all(torch.equal(v,resume['model'][k]) for k,v in state.items())
        assert all(torch.isfinite(v).all() for values in resume['optimizer']['state'].values() for v in values.values() if torch.is_tensor(v))
        report=dict(model=name,complete=True,epochs=args.epochs,steps=steps*args.epochs,optimizer_updates=updates,weights=str(weights),
                    weights_sha256=sha(weights),resume_sha256=sha(resume_path),validation_before=before,
                    validation_after=after,training_seconds=time.time()-started,wandb_url=run.url)
        run.summary.update(report)
        run.finish()
        write(out/'completion.json',report);completions.append(report)
        del state,fresh,resume,model,optimizer,loader
        native.model=None;native.optimizer=None
        gc.collect();torch.cuda.empty_cache()
    write(ROOT/'completion.json',dict(complete=True,models=completions,pod_id='g1ol0tfb8yk3ik',timestamp=time.time()))
    phase('complete_terminating_pod')
    # Credentials live only on the container disk, never in the persistent artifact directory.
    import requests
    key=(RUNTIME/'runpod.key').read_text().strip()
    session=requests.Session();session.headers['Authorization']='Bearer '+key
    pod_id='g1ol0tfb8yk3ik'
    p=session.get('https://api.runpod.io/v2/pods/'+pod_id,timeout=30);p.raise_for_status()
    info=p.json()
    assert info['name']=='minimind-pretrain-rtx5090-20261023'
    assert info['mounts']['global'][0]['volumeId']=='cmuxpmy6p000007l12uzr990b'
    write(ROOT/'cleanup.json',dict(pod_id=pod_id,action='terminate',authorized=True,requested_at=time.time(),global_volume_preserved=True))
    for attempt in range(5):
        r=session.delete('https://api.runpod.io/v2/pods/'+pod_id,timeout=30)
        if r.status_code in [204,404]:return
        if r.status_code not in [429,500,502,503,504]:r.raise_for_status()
        time.sleep(2**attempt)
    raise RuntimeError('Pod termination failed; manual cleanup required')

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        phase('failed',error=type(exc).__name__,detail=str(exc))
        raise
