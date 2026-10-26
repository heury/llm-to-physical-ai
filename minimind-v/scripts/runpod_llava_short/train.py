"""One alignment epoch then one SFT epoch; immutable checkpoints and final evaluation."""
import io, math, random, shutil, sys, tarfile, traceback, zipfile
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from common import *
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from model.model_vlm import MiniMindVLM,VLMConfig

class Data(Dataset):
    def __init__(self,stage,split,processor):
        self.stage=stage;self.processor=processor
        self.indices=np.load(LOCAL/f'{stage}-{split}-images.npy',mmap_mode='r')
        self.data=np.memmap(LOCAL/f'{stage}-{split}.bin',dtype=np.int16,mode='r',shape=(len(self.indices),2,LENGTH))
        self.images=json.loads((ROOT/'data'/f'{stage}-images.json').read_text());self.archive=None
    def __len__(self):return len(self.indices)
    def image(self,index):
        if self.archive is None:self.archive=zipfile.ZipFile(LOCAL/f'{self.stage}.zip')
        with Image.open(io.BytesIO(self.archive.read(self.images[index]['member']))) as im:
            return ImageOps.exif_transpose(im).convert('RGB')
    def __getitem__(self,i):
        pix=self.processor(images=self.image(int(self.indices[i])),return_tensors='pt')['pixel_values'][0]
        return torch.from_numpy(self.data[i,0].astype(np.int64)),torch.from_numpy(self.data[i,1].astype(np.int64)),pix

def loader(data,shuffle=False):
    return DataLoader(data,batch_size=SETTINGS['batch_size'],shuffle=shuffle,generator=torch.Generator().manual_seed(42),num_workers=4,pin_memory=True,persistent_workers=True)

def forward(model,batch,black=None):
    x,y,p=[v.cuda(non_blocking=True) for v in batch]
    if black is not None:p=black.expand(p.shape[0],-1,-1,-1)
    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x,labels=y,pixel_values={'pixel_values':p})
    return out.loss,int((y[:,1:]!=-100).sum())

@torch.inference_mode()
def evaluate(model,dl,black=False):
    model.eval();total=targets=0
    blank=model.processor(images=Image.new('RGB',(256,256),'black'),return_tensors='pt')['pixel_values'].cuda() if black else None
    for batch in dl:
        loss,n=forward(model,batch,blank);assert torch.isfinite(loss);total+=float(loss)*n;targets+=n
    assert targets>0
    return dict(loss=total/targets,perplexity=math.exp(total/targets),assistant_tokens=targets,windows=len(dl.dataset),image_condition='black' if black else 'original')

def log(values):
    values=dict(values,timestamp=time.time())
    with (ROOT/'metrics.jsonl').open('a') as f:f.write(json.dumps(values)+'\n')

def checkpoint(model,opt,stage,step,updates):
    # Stage/step filenames are immutable; verify mounted copy before reporting success.
    stem=f'{stage}-{step:07d}'
    weights={k:v.detach().cpu().half() for k,v in model.state_dict().items() if not k.startswith('vision_encoder.')}
    assert all(torch.isfinite(v).all() for v in weights.values())
    local=LOCAL/f'{stem}.pth';torch.save(weights,local)
    dest=ROOT/'out'/local.name;shutil.copyfile(local,dest);assert sha(local)==sha(dest)
    resume=LOCAL/f'{stage}-resume.pth'
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('vision_encoder.')},optimizer=opt.state_dict(),stage=stage,step=step,updates=updates,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),resume)
    rd=ROOT/'checkpoints'/f'{stem}-resume.pth';shutil.copyfile(resume,rd);assert sha(resume)==sha(rd)
    write(ROOT/'checkpoint.json',dict(stage=stage,step=step,updates=updates,weights=str(dest),sha256=sha(dest)))
    # Keep latest two per stage plus final; minimize persistent checkpoint storage.
    old=sorted((ROOT/'checkpoints').glob(f'{stage}-*-resume.pth'))
    for p in old[:-2]:p.unlink()
    old=sorted((ROOT/'out').glob(f'{stage}-*.pth'))
    for p in old[:-2]:p.unlink()
    for p in LOCAL.glob(f'{stage}-[0-9]*.pth'):
        if p!=local:p.unlink()
    return dest

@torch.inference_mode()
def samples(model,tok):
    model.eval();rows=json.loads((ROOT/'raw/sft.json').read_text());split=json.loads((ROOT/'data/sft-record-splits.json').read_text())['test']
    images=json.loads((ROOT/'data/sft-images.json').read_text());members={r['key']:r['member'] for r in images};answers=[];used=set()
    with zipfile.ZipFile(LOCAL/'sft.zip') as z:
        for idx in split:
            row=rows[idx]
            if row['image'] in used:continue
            used.add(row['image']);j=len(answers)
            with Image.open(io.BytesIO(z.read(members[row['image']]))) as im:image=ImageOps.exif_transpose(im).convert('RGB')
            image.save(ROOT/'samples'/f'{j:02d}.jpg',quality=95)
            pixels=model.processor(images=image,return_tensors='pt')['pixel_values'].cuda()
            question=row['conversations'][0]['value']
            prompt=encode(tok,[{'role':'user','content':question.replace('<image>','<|image_pad|>'*64)}],True)
            assert len(prompt)+128<=LENGTH
            x=torch.tensor([prompt],device='cuda');entry=dict(image=f'samples/{j:02d}.jpg',source_image=row['image'],question=question,reference=row['conversations'][1]['value'])
            for mode in ['greedy','sample','black']:
                pix=pixels if mode!='black' else model.processor(images=Image.new('RGB',(256,256),'black'),return_tensors='pt')['pixel_values'].cuda()
                torch.manual_seed(42)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    y=model.generate(x,pixel_values={'pixel_values':pix},max_new_tokens=128,do_sample=mode=='sample',temperature=.8 if mode=='sample' else 1.,top_p=.9 if mode=='sample' else 1.,top_k=40 if mode=='sample' else 0,repetition_penalty=1.,eos_token_id=2,use_cache=True)
                new=y[0,len(prompt):].tolist();entry[mode]=dict(response=tok.decode(new,skip_special_tokens=True),generated_tokens=len(new),eos_reached=2 in new)
            answers.append(entry);write(ROOT/'samples.json',answers)
            if len(answers)==8:break
    assert len(answers)==8
    return answers

def train():
    assert (ROOT/'PREPARED.json').exists()
    write(ROOT/'settings.json',SETTINGS)
    assert not (ROOT/'completion.json').exists(),'Do not overwrite a completed run'
    torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    config=VLMConfig(**json.loads((ROOT/'base_config.json').read_text()),image_ids=[12],image_token_len=64)
    config.save_pretrained(ROOT)
    model=MiniMindVLM(config,vision_model_path=str(ROOT/'vision_encoder'))
    assert model.vision_encoder is not None
    incompatible=model.load_state_dict(torch.load(LOCAL/'base.pth',map_location='cpu',weights_only=True),strict=False)
    assert not incompatible.unexpected_keys
    assert all(n.startswith(('vision_encoder.','vision_proj.')) for n in incompatible.missing_keys)
    model.cuda();tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True)
    val={s:loader(Data(s,'validation',model.processor)) for s in ['pretrain','sft']}
    status('baseline_evaluation');before={s:evaluate(model,v) for s,v in val.items()};write(ROOT/'evaluation_before.json',before)
    started=time.time();global_updates=0;stages={}
    for stage,lr in [('pretrain',SETTINGS['alignment_lr']),('sft',SETTINGS['sft_lr'])]:
        for name,p in model.named_parameters():
            p.requires_grad_(name.startswith('vision_proj.') or (stage=='sft' and name.startswith(('model.layers.0.','model.layers.7.'))))
        trainable=[p for p in model.parameters() if p.requires_grad];frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
        opt=torch.optim.AdamW(trainable,lr=lr,weight_decay=.01);dl=loader(Data(stage,'train',model.processor));steps=len(dl);updates=0;accum=SETTINGS['accumulation_steps']
        model.train();model.vision_encoder.eval();batch=next(iter(dl));loss,_=forward(model,batch);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        write(ROOT/f'smoke_{stage}.json',dict(loss=float(loss),grad_norm=float(norm),trainable_parameters=sum(p.numel() for p in trainable),peak_cuda_bytes=torch.cuda.max_memory_allocated()))
        opt.zero_grad(set_to_none=True);dl.generator.manual_seed(42);stage_start=time.time()
        for step,batch in enumerate(dl,1):
            model.train();model.vision_encoder.eval()
            group=min(accum,steps-((step-1)//accum)*accum)
            rate=lr*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/steps)))
            for g in opt.param_groups:g['lr']=rate
            loss,_=forward(model,batch);assert torch.isfinite(loss);(loss/group).backward()
            if step%accum==0 or step==steps:
                norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);updates+=1;global_updates+=1
            if step==1 or step%20==0 or step==steps:
                elapsed=time.time()-stage_start;values=dict(stage=stage,step=step,steps=steps,optimizer_updates=updates,global_updates=global_updates,loss=float(loss),learning_rate=rate,elapsed_seconds=elapsed,stage_eta_hours=elapsed/step*(steps-step)/3600)
                log(values);status('training',**values)
            if (step%accum==0 and (updates==1 or updates%500==0)) or step==steps:
                final=checkpoint(model,opt,stage,step,updates)
        assert updates==math.ceil(steps/accum)
        assert all(torch.equal(p.detach().cpu(),frozen[n]) for n,p in model.named_parameters() if n in frozen)
        del frozen
        stages[stage]=dict(steps=steps,updates=updates,seconds=time.time()-stage_start,validation=evaluate(model,val[stage]))
        write(ROOT/f'evaluation_{stage}.json',stages[stage]);log({stage+'_validation_loss':stages[stage]['validation']['loss'],'global_updates':global_updates})
    status('final_evaluation')
    # Evaluate the reloaded exported FP16 weights, not only the in-memory FP32 trainer.
    state=torch.load(final,map_location='cpu',weights_only=True)
    missing=model.load_state_dict(state,strict=False)
    assert not missing.unexpected_keys and all(k.startswith('vision_encoder.') for k in missing.missing_keys)
    assert all(torch.equal(model.state_dict()[k].cpu().half(),v) for k,v in state.items())
    target=ROOT/'out/sft_vlm_768.pth';shutil.copyfile(final,target);assert sha(final)==sha(target)
    test=loader(Data('sft','test',model.processor))
    report=dict(validation=evaluate(model,val['sft']),test=evaluate(model,test),test_black_image=evaluate(model,test,True),stages=stages,seconds=time.time()-started,finite_weights=all(torch.isfinite(v).all().item() for v in state.values()),strict_nonvision_reload=True,weights_sha256=sha(target),global_updates=global_updates)
    samples(model,tok);write(ROOT/'evaluation_after.json',report)
    log({'complete':True,'test_loss':report['test']['loss'],'test_black_loss':report['test_black_image']['loss'],'global_updates':global_updates})
    write(ROOT/'completion.json',dict(complete=True,report=report,weights=str(target),timestamp=time.time()))
    package()
    status('complete_awaiting_export')

def package():
    bundle=LOCAL/'bundle';bundle.mkdir(exist_ok=True)
    for name in ['tokenizer','vision_encoder','samples']:
        shutil.copytree(ROOT/name,bundle/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.cache'))
    shutil.copyfile(ROOT/'out/sft_vlm_768.pth',bundle/'sft_vlm_768.pth')
    for p in ROOT.glob('*.json'):shutil.copyfile(p,bundle/p.name)
    shutil.copytree(Path(__file__).resolve().parents[2]/'model',bundle/'model',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(Path(__file__).parent,bundle/'training_source',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    write(bundle/'SHA256.json',{str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file() and p.name!='SHA256.json'})
    archive=LOCAL/'artifacts.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in bundle.iterdir():t.add(p,arcname=p.name)
    shutil.copyfile(archive,ROOT/'artifacts.tar.gz');assert sha(archive)==sha(ROOT/'artifacts.tar.gz')
    write(ROOT/'export_ready.json',dict(path=str(archive),sha256=sha(archive)))

if __name__=='__main__':
    try:train()
    except BaseException:
        status('failed',traceback=traceback.format_exc());raise
