"""Three-epoch supervised training on the explicitly requested Hermes file."""
import json,math,random,sys,time,hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from trainer.train_agent import parse_tool_calls
from data import normalize,examples,SOURCE,REVISION,FILE
from run import save

def run():
    torch.set_num_threads(4);torch.manual_seed(42);random.seed(42)
    tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer')
    path=Path(hf_hub_download(SOURCE,FILE,repo_type='dataset',revision=REVISION))
    raw=json.loads(path.read_text());train=[];valid=[];normalized=[];rejected=[];skipped=0
    for row in raw:
        try:r=normalize(row);xs,n=examples(r,tok)
        except (ValueError,TypeError,KeyError) as e:
            rejected.append({'id':row.get('id'),'reason':str(e)});continue
        skipped+=n;normalized.append(r)
        (valid if r['validation'] else train).extend(xs)
    assert len(raw)==1893 and train and valid
    manifest={'source':SOURCE,'revision':REVISION,'file':FILE,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_rows':len(raw),'train_conversations':sum(not r['validation'] for r in normalized),
        'validation_conversations':sum(r['validation'] for r in normalized),
        'train_examples':len(train),'validation_examples':len(valid),'overlength_examples_skipped':skipped,
        'rejected':rejected,'max_length':4096,'epochs':3,'batch_size':1,'accumulation_steps':8,
        'learning_rate':2e-5,'seed':42,'train_post_tool_examples':sum(x['post_tool'] for x in train),
        'validation_post_tool_examples':sum(x['post_tool'] for x in valid)}
    save('sft_manifest.json',manifest)
    cfg=MiniMindConfig.from_pretrained(ROOT/'base');model=MiniMindForCausalLM(cfg).cuda()
    initial=torch.load(ROOT/'out/full_sft_768.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(initial,strict=True);del initial
    torch.save(model.state_dict(),ROOT/'out/base_model_768.pth')
    def loss_for(ex):
        ids=torch.tensor([ex['ids']],device='cuda');labels=torch.tensor([ex['labels']],device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(ids).logits
            loss=F.cross_entropy(logits[:,:-1].reshape(-1,cfg.vocab_size).float(),labels[:,1:].reshape(-1),ignore_index=-100)
        return loss,int((labels[:,1:]!=-100).sum())
    def validation():
        model.eval();total=0.;tokens=0
        with torch.no_grad():
            for x in valid:
                loss,n=loss_for(x);total+=loss.item()*n;tokens+=n
        return total/tokens
    def sample_eval():
        from evaluate import GreedyEngine
        engine=GreedyEngine('http://127.0.0.1:8998',str(ROOT/'sglang_policy'),str(ROOT/'sft_eval_policy'))
        engine.update_policy(model)
        selected=[x for x in valid if not x['post_tool']][:16]+[x for x in valid if x['post_tool']][:16]
        report=[]
        for ex in selected:
            ids=torch.tensor([ex['prompt']]);res=engine.rollout(ids,torch.ones_like(ids),1,256)
            answer=res.completions[0];expected=parse_tool_calls(ex['target']);got=parse_tool_calls(answer)
            report.append({'id':ex['id'],'post_tool':ex['post_tool'],'expected':ex['target'],'generated':answer,
                'exact_call_match':sorted(json.dumps(x,sort_keys=True) for x in expected)==sorted(json.dumps(x,sort_keys=True) for x in got) if expected else None,
                'post_tool_completed_without_call':bool(answer.strip()) and not got and '<tool_call>' not in answer if ex['post_tool'] else None})
        return report
    save('status.json',{'stage':'sft_baseline','updated_at':time.time()})
    before_loss=validation();before=sample_eval();save('sft_before.json',{'validation_loss':before_loss,'samples':before})
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    total_steps=3*math.ceil(len(train)/8);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,total_steps,eta_min=2e-6)
    start=time.time();step=0;best=float('inf');history=[]
    for epoch in range(3):
        model.train();order=list(train);random.shuffle(order)
        for offset in range(0,len(order),8):
            group=order[offset:offset+8];opt.zero_grad();mean=0.
            for ex in group:
                loss,_=loss_for(ex);assert torch.isfinite(loss)
                (loss/len(group)).backward();mean+=loss.item()/len(group)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();scheduler.step();step+=1
            if step%10==0 or step==total_steps:
                save('status.json',{'stage':'sft_training','epoch':epoch+1,'step':step,'total_steps':total_steps,'loss':mean,'elapsed_seconds':time.time()-start,'updated_at':time.time()})
                print(json.dumps({'sft_step':step,'total':total_steps,'loss':mean}),flush=True)
        score=validation();history.append({'epoch':epoch+1,'validation_loss':score})
        if score<best:
            best=score;torch.save({k:v.half().cpu() for k,v in model.state_dict().items()},ROOT/'out/hermes_sft_768.pth')
    state=torch.load(ROOT/'out/hermes_sft_768.pth',map_location='cpu',weights_only=True)
    assert all(torch.isfinite(v).all() for v in state.values())
    model.load_state_dict(state,strict=True);after=sample_eval()
    save('sft_after.json',{'validation_loss':best,'samples':after,'epochs':history})
    torch.save(state,ROOT/'out/full_sft_768.pth') # exact seed used by Agent RL and reference model
    export=ROOT/'export_sft';model=model.cpu().half().eval()
    MiniMindConfig.register_for_auto_class();MiniMindForCausalLM.register_for_auto_class('AutoModelForCausalLM')
    model.save_pretrained(export,safe_serialization=True);tok.save_pretrained(export)
    from transformers import AutoModelForCausalLM
    loaded=AutoModelForCausalLM.from_pretrained(export,trust_remote_code=True,torch_dtype=torch.float16)
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
    save('sft_completion.json',{'complete':True,'optimizer_updates':step,'epochs':3,'training_seconds':time.time()-start,
        'finite_weights':True,'strict_reload_verified':True,'model_sha256':hashlib.sha256((export/'model.safetensors').read_bytes()).hexdigest(),
        'before_validation_loss':before_loss,'after_validation_loss':best})
    del model,loaded,opt;torch.cuda.empty_cache()
    return manifest
