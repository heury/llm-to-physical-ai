import hashlib,json,math,os,random,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from dataset.lm_dataset import DPODataset
from trainer.train_dpo import logits_to_log_probs,dpo_loss

root=Path(os.environ['DPO_ROOT']);runtime=Path(os.environ['DPO_RUNTIME']);phase=os.environ.get('DPO_PHASE','before')
torch.set_num_threads(4);torch.manual_seed(42);random.seed(42)
tok=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True)
base=Path('/shared-volume/minimind-sft-ultrachat-200k-20261020/out/full_sft_768.pth')
def load(weight):
    model=MiniMindForCausalLM(MiniMindConfig(hidden_size=768,num_hidden_layers=8))
    state=torch.load(weight,map_location='cpu',weights_only=True)
    assert all(torch.isfinite(v).all() for v in state.values())
    model.load_state_dict(state,strict=True)
    return model.cuda().eval()
ref=load(base)
if phase=='smoke':
    policy=load(base).train()
    ds=DPODataset(str(runtime/'train_prefs.jsonl'),tok,max_length=1024)
    batch=next(iter(DataLoader(ds,batch_size=32)))
    x=torch.cat([batch['x_chosen'],batch['x_rejected']]).cuda();y=torch.cat([batch['y_chosen'],batch['y_rejected']]).cuda();mask=torch.cat([batch['mask_chosen'],batch['mask_rejected']]).cuda()
    assert (mask.sum(1)>=32).all()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        with torch.no_grad():r=logits_to_log_probs(ref(x).logits,y)
        p=logits_to_log_probs(policy(x).logits,y)
        loss=dpo_loss(r,p,mask,0.1)
    assert abs(loss.item()-math.log(2))<1e-4
    loss.backward();grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.)
    assert torch.isfinite(grad) and grad>0
    report=dict(initial_loss=loss.item(),expected_loss=math.log(2),grad_norm=float(grad),peak_cuda_bytes=torch.cuda.max_memory_allocated(),pairs=32,optimizer_update=False)
else:
    policy=ref if phase=='before' else load(root/'out/dpo_768.pth')
    ds=DPODataset(str(runtime/'validation.jsonl'),tok,max_length=1024)
    total=correct=margin_sum=raw_correct=chosen_nll=chosen_tokens=0;loss_sum=0
    for batch in DataLoader(ds,batch_size=16,num_workers=0):
        x=torch.cat([batch['x_chosen'],batch['x_rejected']]).cuda();y=torch.cat([batch['y_chosen'],batch['y_rejected']]).cuda();mask=torch.cat([batch['mask_chosen'],batch['mask_rejected']]).cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            r=logits_to_log_probs(ref(x).logits,y)
            p=r if phase=='before' else logits_to_log_probs(policy(x).logits,y)
            loss=dpo_loss(r,p,mask,0.1)
            rewards=0.1*((p-r)*mask).sum(1);a,b=rewards.chunk(2);m=a-b
            pc,pr=((p*mask).sum(1)).chunk(2)
            n=len(a);loss_sum+=loss.item()*n;total+=n;correct+=(m>0).sum().item();margin_sum+=m.sum().item();raw_correct+=(pc>pr).sum().item()
            chosen_nll-=pc.sum().item();chosen_tokens+=mask[:n].sum().item()
    report=dict(phase=phase,validation_pairs=total,dpo_loss=loss_sum/total,reward_accuracy=correct/total,reward_margin=margin_sum/total,raw_sequence_preference_accuracy=raw_correct/total,chosen_token_nll=chosen_nll/chosen_tokens,beta=0.1,note='Reward accuracy at baseline is zero because policy equals reference; raw sequence accuracy is length-sensitive.')
    assert math.isfinite(report['dpo_loss'])
    if phase=='after':
        checkpoint=torch.load(root/'checkpoints/dpo_768_resume.pth',map_location='cpu',weights_only=True)
        manifest=json.loads((root/'dataset_manifest.json').read_text())
        assert checkpoint['epoch']==0 and checkpoint['step']==manifest['steps_per_epoch']
        report['wandb_id']=checkpoint['wandb_id'];assert report['wandb_id'];del checkpoint
        with (root/'out/dpo_768.pth').open('rb') as f:report['checkpoint_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
        report['samples']=[];policy.bfloat16()
        for prompt in ['Explain photosynthesis in two sentences.','How do I calculate the area of a circle?','Write a Python function that adds two numbers.','List three tips for learning a language.']:
            text=tok.apply_chat_template([dict(role='system',content='You are a helpful assistant.'),dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True,open_thinking=False)
            inputs=tok(text,return_tensors='pt').input_ids.cuda()
            with torch.inference_mode():out=policy.generate(inputs=inputs,max_new_tokens=160,do_sample=False,repetition_penalty=1.05,eos_token_id=tok.eos_token_id)
            report['samples'].append(dict(prompt=prompt,response=tok.decode(out[0,inputs.shape[1]:],skip_special_tokens=True)))
(root/('evaluation_'+phase+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False),flush=True)
