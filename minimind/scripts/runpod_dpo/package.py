import hashlib,json,os,shutil,tarfile
from pathlib import Path
import wandb
root=Path(os.environ['DPO_ROOT']);runtime=Path(os.environ['DPO_RUNTIME'])
after=json.loads((root/'evaluation_after.json').read_text())
try:
    run=wandb.Api().run('mind4u/MiniMind-UltraFeedback-DPO/'+after['wandb_id'])
    for k in ['dpo_loss','reward_accuracy','reward_margin','chosen_token_nll']:run.summary['validation_'+k]=after[k]
    run.summary.update()
except Exception as e:(root/'wandb_summary_warning.txt').write_text(type(e).__name__)
bundle=runtime/'artifacts';bundle.mkdir(exist_ok=True)
shutil.copyfile(root/'out/dpo_768.pth',bundle/'dpo_768.pth')
for n in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']:shutil.copyfile(root/'tokenizer'/n,bundle/n)
for n in ['config.json','dataset_manifest.json','training_settings.json','evaluation_before.json','evaluation_after.json','evaluation_smoke.json','requirements-resolved.txt']:shutil.copyfile(root/n,bundle/n)
shutil.copyfile(runtime/'repo/model/model_minimind.py',bundle/'model_minimind.py')
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()}
(bundle/'SHA256.json').write_text(json.dumps(hashes,indent=2))
archive=runtime/'dpo-artifacts.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(bundle.iterdir()):tar.add(p,arcname=p.name)
shutil.copyfile(archive,root/'dpo-artifacts.tar.gz');print('PACKAGED',archive,flush=True)
