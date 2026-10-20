import fcntl,os,runpy,traceback
from pathlib import Path
root=Path('/shared-volume/minimind-v-sharegpt4v-20261020')
runtime=Path('/tmp/minimind-v-sharegpt4v-20261020')
lock=(runtime/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (root/'SUCCESS').exists()
os.environ.update(TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',WANDB_MODE='online',HF_HUB_DISABLE_XET='1',OMP_NUM_THREADS='4',PYTHONPATH=str(runtime/'repo'))
key=runtime/'wandb.key';os.environ['WANDB_API_KEY']=key.read_text().strip();key.unlink()
import sys
sys.path.insert(0,str(runtime/'repo'))
code=1
try:
 runpy.run_path(str(runtime/'repo/scripts/runpod_sharegpt4v/run.py'),run_name='__main__');code=0
except BaseException:traceback.print_exc()
finally:(root/'exit_code').write_text(str(code)+'\n')
raise SystemExit(code)
