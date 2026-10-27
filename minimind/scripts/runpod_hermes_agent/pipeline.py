import json,time,traceback
from pathlib import Path
from run import save,main as rl_main
from sft import run as sft_run
ROOT=Path(__file__).resolve().parents[2]
try:
    sft_run()
    save('status.json',{'stage':'sft_ready_to_publish','updated_at':time.time()})
    start=time.time()
    while not (ROOT/'sft_published.json').exists():
        if time.time()-start>3600:raise RuntimeError('SFT publication acknowledgement timeout')
        time.sleep(15)
    rl_main()
except Exception as exc:
    save('status.json',{'stage':'failed','error':str(exc),'traceback':traceback.format_exc(),'updated_at':time.time()})
    raise
