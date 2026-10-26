"""Recover only HF export after Global Volume rejects chmod; never retrain."""
import json,hashlib,math,shutil,time
from pathlib import Path
import torch,numpy as np
from transformers import AutoModelForCausalLM,AutoTokenizer
from train import score,write
root=Path('/shared-volume/minimind-dense-dpo-short-20261025');work=root/'results';device='cuda:0';torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
history=json.loads((work/'validation_history.json').read_text());assert len(history)==2
logs=[json.loads(x) for x in (work/'metrics.jsonl').read_text().splitlines()];assert len(logs)==170 and logs[-1]['epoch']==2
policy=AutoModelForCausalLM.from_pretrained(root/'base',trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32).to(device).eval()
base={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
policy.load_state_dict(torch.load(work/'best.pth',map_location=device,weights_only=True),strict=True)
a=np.load(root/'data/test.npy');scores,_=score(policy,a,device);np.testing.assert_array_equal(scores,np.load(work/'policy-test.npy'))
changed=sum(int((v.detach().cpu()!=base[k]).sum()) for k,v in policy.named_parameters());delta=math.sqrt(sum(float((v.detach().cpu()-base[k]).square().sum()) for k,v in policy.named_parameters()))
assert changed>0 and delta>0
out=Path('/tmp/dpo-model-export');policy.config.model_version='v1.0';policy.config.finetuning_task='dpo-ultrafeedback-short';policy.config.base_model_name_or_path='caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0';policy.__class__.register_for_auto_class('AutoModelForCausalLM');policy.config.__class__.register_for_auto_class();policy.save_pretrained(out,safe_serialization=True)
tok=AutoTokenizer.from_pretrained(root/'base',local_files_only=True);tok.save_pretrained(out)
restored=AutoModelForCausalLM.from_pretrained(out,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32).to(device).eval()
assert all(torch.equal(v,restored.state_dict()[k]) for k,v in policy.state_dict().items());again,_=score(restored,a,device);np.testing.assert_array_equal(again,scores)
for p in out.rglob('*'):
 if p.is_file():
  dst=root/'model'/p.relative_to(out);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
with (out/'model.safetensors').open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
with (root/'model/model.safetensors').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
summary={'complete':True,'training_epochs':2,'selected_epoch':min(history,key=lambda r:r['validation']['dpo_loss'])['epoch'],'optimizer_steps':170,'training_seconds':logs[-1]['elapsed_seconds'],'training_seconds_measurement':'elapsed through last optimizer update, includes inter-epoch validation; excludes final evaluation/export','before':json.loads((work/'before.json').read_text()),'after':json.loads((work/'after.json').read_text()),'reference_unchanged':True,'reference_check_evidence':'original training assert statehash(ref)==ref_hash passed before samples and failed export','changed_parameter_elements':changed,'parameter_delta_l2':delta,'model_sha256':digest,'export_dtype':'float32','reload_weights_and_scores_exact':True,'all_test_scores_match_before_export':True,'peak_gpu_memory_gb':None,'export_recovered_after_filesystem_chmod_error':True,'completed_at':time.time()}
write(work/'completion.json',summary);write(work/'status.json',{'phase':'complete','timestamp':time.time()});print(json.dumps(summary),flush=True)
