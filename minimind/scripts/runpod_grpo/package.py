import hashlib,json,os,shutil,tarfile
from pathlib import Path
import wandb
root=Path(os.environ['GRPO_ROOT']);runtime=Path(os.environ['GRPO_RUNTIME'])
after=json.loads((root/'evaluation_after.json').read_text())
before=json.loads((root/'evaluation_before.json').read_text())
try:
    run=wandb.Api().run('mind4u/MiniMind-UltraFeedback-GRPO/'+after['wandb_id'])
    for key in ['mean_reward','mean_raw_reward','mean_repetition_penalty','mean_completion_tokens','eos_rate','empty_rate']:
        run.summary['validation_before_'+key]=before[key];run.summary['validation_after_'+key]=after[key]
    run.summary.update()
except Exception as e:(root/'wandb_summary_warning.txt').write_text(type(e).__name__)
bundle=runtime/'artifacts';bundle.mkdir(exist_ok=True)
shutil.copyfile(root/'out/grpo_768.pth',bundle/'grpo_768.pth')
for n in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']:shutil.copyfile(root/'tokenizer'/n,bundle/n)
for n in ['config.json','dataset_manifest.json','training_settings.json','smoke.json','evaluation_before.json','evaluation_after.json','evaluation_before_samples.json','evaluation_after_samples.json','requirements-resolved.txt']:shutil.copyfile(root/n,bundle/n)
shutil.copyfile(runtime/'repo/model/model_minimind.py',bundle/'model_minimind.py')
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()}
(bundle/'SHA256.json').write_text(json.dumps(hashes,indent=2))
archive=runtime/'grpo-artifacts.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(bundle.iterdir()):tar.add(p,arcname=p.name)
shutil.copyfile(archive,root/'grpo-artifacts.tar.gz')
print('PACKAGED',str(archive),flush=True)
