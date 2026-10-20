"""Detached launcher; records failure explicitly and never stops the user's Pod."""
import fcntl
import os
import traceback
from pathlib import Path

runtime = Path('/tmp/minimind-lora-chatdoctor-20261020')
root = Path('/shared-volume/minimind-lora-chatdoctor-20261020')
lock = (runtime/'run.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (root/'SUCCESS').exists()
os.environ.update(PYTHONPATH=str(runtime/'repo'), TOKENIZERS_PARALLELISM='false',
    PYTHONUNBUFFERED='1', WANDB_MODE='online', WANDB_DIR=str(runtime/'wandb'),
    WANDB_CACHE_DIR=str(runtime/'wandb-cache'), OMP_NUM_THREADS='2')
(runtime/'wandb').mkdir(exist_ok=True)
key = runtime/'wandb.key'
code = 1
try:
    os.environ['WANDB_API_KEY'] = key.read_text().strip()
    key.unlink()
    import sys
    sys.path.insert(0, str(runtime/'repo'))
    from run import train
    train()
    code = 0
except BaseException:
    traceback.print_exc()
finally:
    (root/'exit_code').write_text(str(code)+'\n')
raise SystemExit(code)
