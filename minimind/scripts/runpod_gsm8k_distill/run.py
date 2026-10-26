"""Pinned MoE -> Dense soft KD; only released GSM8K training rows train."""
import collections, gc, json, math, os, random, shutil, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import chat_text, assistant_spans, window_bounds, canonical, extract, write, sha, LENGTH, SUFFIX

ROOT=Path(os.environ.get('DISTILL_ROOT','/workspace/moe-dense-gsm8k-distill-20261025'))
SOURCES=json.loads((Path(__file__).parent/'sources.json').read_text())
SETTINGS=dict(epochs=3,batch_size=16,accumulation_steps=4,learning_rate=5e-6,min_learning_rate=5e-7,temperature=1.5,ce_weight=.5,kl_weight=.5,seed=42,weight_decay=.01,grad_clip=1.,max_seq_len=340,stride=256)

def status(phase,**kw):
    row=dict(phase=phase,time=time.time(),**kw);write(ROOT/'status.json',row);print(json.dumps(row),flush=True)
def download(kind,file):
    s=SOURCES[kind]
    return hf_hub_download(s['repo'],file,revision=s['revision'],repo_type='dataset' if kind=='dataset' else 'model')
def losses(s,t,y,temp=1.5):
    mask=y[:,1:]!=-100
    sl=s[:,:-1][mask].float();tl=t[:,:-1][mask].float()
    ce=F.cross_entropy(sl,y[:,1:][mask],reduction='sum')
    kl=F.kl_div(F.log_softmax(sl/temp,-1),F.softmax(tl/temp,-1),reduction='sum')*temp**2
    return ce,kl,mask.sum()
def encode(rows,tok):
    pairs=[];targets=0
    bos=tok('<|im_start|>assistant\n',add_special_tokens=False).input_ids
    eos=tok(tok.eos_token+'\n',add_special_tokens=False).input_ids
    for row in rows:
        msgs=json.loads(row['messages_json']);ids=tok(chat_text(tok,msgs),add_special_tokens=False).input_ids
        labels=np.full(len(ids),-100,dtype=np.int64)
        for a,b in assistant_spans(ids,bos,eos):labels[a:b]=ids[a:b]
        covered=1;written=0
        for start,end in window_bounds(len(ids)):
            y=labels[start:end].copy();y[:max(1,covered-start)]=-100;covered=max(covered,end)
            n=int((y[1:]!=-100).sum())
            if not n:continue
            pair=np.zeros((2,LENGTH),dtype=np.int64);pair[1].fill(-100)
            pair[0,:end-start]=ids[start:end];pair[1,:end-start]=y;pairs.append(pair);written+=n
        assert written==int((labels[1:]!=-100).sum())==row['assistant_targets']
        targets+=written
    return torch.from_numpy(np.stack(pairs)),targets

def prepare(tok):
    hashes=json.loads(Path(download('dataset','SHA256SUMS.json')).read_text())
    def verified(file):
        path=download('dataset',file);assert sha(path)==hashes[file]['sha256'] and Path(path).stat().st_size==hashes[file]['bytes'],file;return path
    # Export recipe writes all GSM8K rows first. Prove complete extraction against manifest.
    raw=pq.read_table(verified('conversations/train/00000.parquet')).to_pylist()
    train=[r for r in raw if r['dataset']=='gsm8k']
    test=pq.read_table(verified('conversations/validation_gsm8k/00000.parquet')).to_pylist()
    manifest=json.loads(Path(verified('provenance/prepared.json')).read_text())
    expected=manifest['stats']['train'];assert len(train)==expected['source_gsm8k_rows']==6851
    assert len(test)==1319 and all(r['dataset']=='gsm8k' for r in test)
    questions={canonical(json.loads(r['messages_json'])[0]['content']) for r in test}
    assert not any(canonical(json.loads(r['messages_json'])[0]['content']) in questions for r in train)
    a,targets=encode(train,tok);v,vt=encode(test,tok)
    assert len(a)==expected['source_gsm8k_windows']==7333 and targets==expected['source_gsm8k_targets']
    write(ROOT/'data.json',dict(train_rows=len(train),train_windows=len(a),train_targets=targets,test_rows=len(test),test_windows=len(v),test_targets=vt,train_source_indices=[r['index'] for r in train],no_exact_test_prompt_overlap=True))
    return a,v,test

def model(kind):
    s=SOURCES[kind];path=snapshot_download(s['repo'],revision=s['revision'])
    m=AutoModelForCausalLM.from_pretrained(path,trust_remote_code=True,torch_dtype=torch.float32).cuda()
    return m

@torch.inference_mode()
def validation(student,teacher,data):
    student.eval();sums=np.zeros(3)
    for pair in data.split(16):
        x,y=pair[:,0].cuda(),pair[:,1].cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            s=student(x).logits;t=teacher(x).logits
        ce,kl,n=losses(s,t,y);sums+=float(ce),float(kl),int(n)
    return dict(ce=sums[0]/sums[2],kl=sums[1]/sums[2],perplexity=math.exp(sums[0]/sums[2]),targets=int(sums[2]))

@torch.inference_mode()
def evaluate(m,tok,rows,label):
    m.eval();groups=collections.defaultdict(list);results=[];start=time.time()
    for row in rows:
        msgs=json.loads(row['messages_json']);gold=extract(msgs[-1]['content']);assert gold is not None
        ids=tok(chat_text(tok,msgs[:-1],True),add_special_tokens=False).input_ids
        item=dict(index=row['index'],gold=gold,prompt_tokens=len(ids))
        if len(ids)>=340:results.append(dict(**item,response='',strict_correct=False,flexible_correct=False,context_overflow=True));continue
        groups[len(ids)].append((item,ids))
    for length,group in sorted(groups.items()):
        for offset in range(0,len(group),32):
            chunk=group[offset:offset+32];x=torch.tensor([ids for _,ids in chunk],device='cuda');budget=min(256,340-length)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                outputs=m.generate(inputs=x,max_new_tokens=budget,do_sample=False,temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,use_cache=True,eos_token_id=tok.eos_token_id)
            for (item,_),ids in zip(chunk,outputs[:,length:].tolist()):
                ended=tok.eos_token_id in ids
                if ended:ids=ids[:ids.index(tok.eos_token_id)+1]
                text=tok.decode(ids,skip_special_tokens=True);strict=extract(text);flex=extract(text,False)
                results.append(dict(**item,response=text,strict_prediction=strict,flexible_prediction=flex,strict_correct=strict==item['gold'],flexible_correct=flex==item['gold'],eos_reached=ended,generated_tokens=len(ids),context_overflow=False))
        write(ROOT/f'generations_{label}.json',results)
        status('evaluate_'+label,evaluated=len(results),total=len(rows),seconds=time.time()-start)
    assert len(results)==1319 and len({r['index'] for r in results})==1319
    strict=sum(r['strict_correct'] for r in results);flex=sum(r['flexible_correct'] for r in results)
    report=dict(test_rows=len(results),strict_correct=strict,strict_accuracy=strict/len(results),flexible_correct=flex,flexible_accuracy=flex/len(results),context_overflow=sum(r['context_overflow'] for r in results),seconds=time.time()-start,settings=dict(do_sample=False,max_new_tokens=256,context_limit=340,zero_shot=True,prompt_suffix=SUFFIX,padding='none; exact length groups'))
    write(ROOT/f'evaluation_{label}.json',report);return report

def main():
    ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    write(ROOT/'sources.json',SOURCES);write(ROOT/'settings.json',SETTINGS)
    status('prepare')
    # Compare full tokenization implementation, not only vocabulary width.
    for f in ['tokenizer.json','tokenizer_config.json']:
        assert Path(download('student',f)).read_bytes()==Path(download('teacher',f)).read_bytes()
    tok=AutoTokenizer.from_pretrained(SOURCES['student']['repo'],revision=SOURCES['student']['revision'])
    train,val,test=prepare(tok);teacher=model('teacher').eval().requires_grad_(False);student=model('student')
    assert teacher.config.vocab_size==student.config.vocab_size==len(tok)==6400
    initial={k:v.cpu().clone() for k,v in student.state_dict().items()}
    # Probe meaningful full-softmax loss and backward before any optimizer update.
    batch=train[:16].cuda();student.train()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        sl=student(batch[:,0]).logits
        with torch.no_grad():tl=teacher(batch[:,0]).logits
    ce,kl,n=losses(sl,tl,batch[:,1]);loss=(ce+kl)/2/n;assert torch.isfinite(loss) and kl>0
    loss.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in student.parameters())
    assert all(p.grad is None for p in teacher.parameters());student.zero_grad(set_to_none=True)
    write(ROOT/'preflight.json',dict(ce=float(ce/n),kl=float(kl/n),loss=float(loss),cuda_backward=True,teacher_frozen=True,tokenizers_identical=True,gpu=torch.cuda.get_device_name(),peak_bytes=torch.cuda.max_memory_allocated()))
    del batch,sl,tl,ce,kl,n,loss
    before=validation(student,teacher,val);write(ROOT/'validation_before.json',before)
    opt=torch.optim.AdamW(student.parameters(),lr=5e-6,weight_decay=.01)
    updates_per_epoch=math.ceil(len(train)/64);total=3*updates_per_epoch;step=0;started=time.time();history=[]
    for epoch in range(3):
        student.train();order=torch.randperm(len(train),generator=torch.Generator().manual_seed(42+epoch));totals=np.zeros(3)
        for start in range(0,len(order),64):
            selected=train[order[start:start+64]];count=int((selected[:,1,1:]!=-100).sum())
            lr=5e-7+.5*(5e-6-5e-7)*(1+math.cos(math.pi*step/max(total-1,1)))
            for group in opt.param_groups:group['lr']=lr
            opt.zero_grad(set_to_none=True);values=np.zeros(3)
            for pair in selected.split(16):
                x,y=pair[:,0].cuda(),pair[:,1].cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    sl=student(x).logits
                    with torch.no_grad():tl=teacher(x).logits
                ce,kl,n=losses(sl,tl,y);loss=(ce+kl)*.5/count
                assert torch.isfinite(loss);loss.backward();values+=float(ce.detach()),float(kl.detach()),int(n)
                del sl,tl,loss,ce,kl,x,y
            norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True);opt.step();step+=1;totals+=values
            row=dict(epoch=epoch+1,update=step,ce=values[0]/values[2],kl=values[1]/values[2],loss=(values[0]+values[1])*.5/values[2],lr=lr,grad_norm=float(norm),seconds=time.time()-started)
            with (ROOT/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step%10==0 or start+64>=len(order):status('train',**row,total_updates=total)
        history.append(dict(epoch=epoch+1,ce=totals[0]/totals[2],kl=totals[1]/totals[2],targets=int(totals[2])))
        tmp=ROOT/'resume.tmp';torch.save(dict(model=student.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,updates=step),tmp);tmp.replace(ROOT/'resume.pt')
        write(ROOT/'training.json',dict(epochs=epoch+1,updates=step,history=history,seconds=time.time()-started))
    assert step==total and any(not torch.equal(v.cpu(),initial[k]) for k,v in student.state_dict().items())
    assert all(torch.isfinite(v).all() for v in student.state_dict().values())
    folder=ROOT/'model';student.config.finetuning_task='gsm8k-soft-distillation';student.config.model_version='v1.0'
    # Export FP16, then reload exactly the artifact evaluated/published.
    student.half().save_pretrained(folder,safe_serialization=True);tok.save_pretrained(folder)
    del student,opt;gc.collect();torch.cuda.empty_cache()
    student=AutoModelForCausalLM.from_pretrained(folder,trust_remote_code=True,torch_dtype=torch.float32).cuda().eval()
    resume=torch.load(ROOT/'resume.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v.cpu().half(),resume['model'][k].half()) for k,v in student.state_dict().items());del resume,initial
    after=validation(student,teacher,val);write(ROOT/'validation_after.json',after)
    del teacher;gc.collect();torch.cuda.empty_cache()
    after_eval=evaluate(student,tok,test,'distilled');del student;gc.collect();torch.cuda.empty_cache()
    baseline=model('student').eval();before_eval=evaluate(baseline,tok,test,'baseline')
    result=dict(complete=True,epochs=3,updates=step,training=json.loads((ROOT/'training.json').read_text()),validation_before=before,validation_after=after,baseline=before_eval,distilled=after_eval,checkpoint_reload_verified=True,teacher_frozen=True)
    write(ROOT/'completion.json',result)
    for name in ['sources.json','settings.json','data.json','training.json','validation_before.json','validation_after.json','evaluation_baseline.json','evaluation_distilled.json','completion.json','metrics.jsonl','preflight.json','generations_baseline.json','generations_distilled.json']:
        shutil.copyfile(ROOT/name,folder/name)
    write(ROOT/'artifact_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in folder.rglob('*') if p.is_file()})
    status('complete',updates=step,baseline=before_eval['strict_accuracy'],distilled=after_eval['strict_accuracy'])

if __name__=='__main__':
    try:main()
    except Exception as e:
        status('failed',error=repr(e));raise
