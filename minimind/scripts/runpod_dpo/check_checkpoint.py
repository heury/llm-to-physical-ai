import hashlib,json,os
from pathlib import Path
import torch
root=Path(os.environ.get('DPO_ROOT','/shared-volume/minimind-dpo-ultrafeedback-20261020'))
torch.set_num_threads(4)
p=root/'checkpoints/dpo_768_resume.pth'
c=torch.load(p,map_location='cpu',weights_only=True)
assert c['epoch']==0 and c['step']>=250
assert c['wandb_id']=='ibg9ezt3'
assert c['optimizer']['state']
assert all(torch.isfinite(t).all() for t in c['model'].values())
for state in c['optimizer']['state'].values():
    for v in state.values():
        if torch.is_tensor(v):assert torch.isfinite(v).all()
report=dict(epoch=c['epoch'],step=c['step'],wandb_id=c['wandb_id'],model_tensors_finite=True,optimizer_tensors_finite=True,resume_bytes=p.stat().st_size)
(root/'first_checkpoint_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
