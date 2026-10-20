import os
from pathlib import Path
runtime=Path('/tmp/minimind-dpo-ultrafeedback-20261020')
p=runtime/'wandb.key';os.environ['WANDB_API_KEY']=p.read_text().strip();p.unlink()
os.execv('/bin/bash',['bash',str(runtime/'launch.sh')])
