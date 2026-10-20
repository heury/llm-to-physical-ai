import os,sys,runpy,traceback,fcntl
from pathlib import Path
root=Path('/shared-volume/minimind-o-voiceassistant-20261020');root.mkdir(parents=True,exist_ok=True)
runtime=Path('/tmp/minimind-o-voiceassistant-20261020')
lock=(runtime/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
os.environ.update(PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_XET='1',WANDB_MODE='offline',OMP_NUM_THREADS='4')
sys.path[:0]=[str(runtime/'repo'),str(runtime/'repo/scripts/runpod_voiceassistant')]
code=1
try:runpy.run_path(str(runtime/'repo/scripts/runpod_voiceassistant/run.py'),run_name='__main__');code=0
except BaseException:traceback.print_exc()
finally:(root/'exit_code').write_text(str(code)+'\n')
raise SystemExit(code)
