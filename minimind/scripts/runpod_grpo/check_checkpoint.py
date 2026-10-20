import json,os
from pathlib import Path
import torch
root=Path(os.environ.get('GRPO_ROOT','/shared-volume/minimind-grpo-ultrafeedback-20261020'))
torch.set_num_threads(4)
c=torch.load(root/'checkpoints/grpo_768_resume.pth',map_location='cpu',weights_only=True)
run=json.loads((root/'wandb_run.json').read_text())
assert c['epoch']==0 and c['step']>=1 and c['wandb_id']==run['id']
assert all(torch.isfinite(v).all() for v in c['model'].values())
steps=[]
for state in c['optimizer']['state'].values():
    steps.append(int(state['step']))
    for value in state.values():
        if torch.is_tensor(value):assert torch.isfinite(value).all()
assert steps and min(steps)==max(steps)==c['step']
base=torch.load('/shared-volume/minimind-dpo-ultrafeedback-20261020/out/dpo_768.pth',map_location='cpu',weights_only=True)
changed=sum(not torch.equal(v,base[k]) for k,v in c['model'].items());assert changed>0
report=dict(step=c['step'],epoch=1,wandb_id=c['wandb_id'],model_tensors_finite=True,optimizer_tensors_finite=True,optimizer_updates=max(steps),changed_tensors_vs_dpo=changed,resume_bytes=(root/'checkpoints/grpo_768_resume.pth').stat().st_size)
(root/'first_checkpoint_validation.json').write_text(json.dumps(report,indent=2))
print('GRPO_FIRST_CHECK:'+json.dumps(report),flush=True)
