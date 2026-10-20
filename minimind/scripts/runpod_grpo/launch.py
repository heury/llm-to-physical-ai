"""Detached job: baseline → full-epoch GRPO → final checks/eval → package."""
import fcntl,json,os,subprocess,sys,traceback
from pathlib import Path
runtime=Path('/tmp/minimind-grpo-ultrafeedback-20261020')
root=Path('/shared-volume/minimind-grpo-ultrafeedback-20261020')
lock=(runtime/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (root/'SUCCESS').exists()
os.environ.update(GRPO_ROOT=str(root),GRPO_RUNTIME=str(runtime),HF_HOME=str(runtime/'hf-cache'),HF_DATASETS_CACHE=str(runtime/'datasets-cache'),WANDB_DIR=str(runtime/'wandb'),WANDB_CACHE_DIR=str(runtime/'wandb-cache'),WANDB_MODE='online',PYTHONPATH=str(runtime/'repo'),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
(runtime/'wandb').mkdir(exist_ok=True)
key=runtime/'wandb.key'
python='/tmp/minimind-checkpoint-test-20261020/venv/bin/python'
def mark(phase):
    (root/"phase.json").write_text(json.dumps(dict(phase=phase)))
def run(command,log,phase=None,cwd=None):
    env=dict(os.environ)
    if phase:env['GRPO_PHASE']=phase
    with (root/log).open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=cwd,check=True)
code=1
try:
    os.environ['WANDB_API_KEY']=key.read_text().strip();key.unlink()
    assert (root/'smoke_exit_code').read_text().strip()=='0'
    smoke=json.loads((root/'smoke.json').read_text());assert smoke['metrics']['ratio_error_mean']<1e-3
    settings=json.loads((root/'training_settings.json').read_text())
    with (root/'requirements-resolved.txt').open('w') as f:subprocess.run([python,'-m','pip','freeze'],stdout=f,check=True)
    if not (root/'evaluation_before.json').exists():
        mark('baseline_evaluation')
        run([python,str(runtime/'evaluate.py')],'evaluation_before.log',phase='before')
    else:
        baseline=json.loads((root/'evaluation_before.json').read_text())
        assert baseline['phase']=='before' and baseline['validation_prompts']==128
    command=[python,'train_grpo.py']
    for key in ['epochs','batch_size','num_generations','accumulation_steps','learning_rate','beta','epsilon','loss_type','max_seq_len','max_gen_len','dtype','thinking_ratio','save_interval','log_interval','hidden_size','num_hidden_layers']:
        command.extend(['--'+key,str(settings[key])])
    command.extend(['--data_path',str(runtime/'train_prefs.jsonl'),'--num_workers','4','--from_weight','dpo','--save_dir',str(root/'out'),'--save_weight','grpo','--reward_model_path',str(runtime/'reward_model'),'--rollout_engine','torch','--use_wandb','--wandb_project','MiniMind-UltraFeedback-GRPO'])
    assert (runtime/'train_prefs.jsonl').is_file() and command[command.index('--data_path')+1]==str(runtime/'train_prefs.jsonl')
    mark('training')
    run(command,'training.log',cwd=runtime/'repo/trainer')
    mark('final_evaluation')
    run([python,str(runtime/'evaluate.py')],'evaluation_after.log',phase='after')
    mark('packaging')
    run([python,str(runtime/'package.py')],'package.log')
    (root/'SUCCESS').write_text('completed\n');mark('complete');code=0
except BaseException:
    traceback.print_exc()
finally:(root/'exit_code').write_text(str(code)+'\n')
sys.exit(code)
