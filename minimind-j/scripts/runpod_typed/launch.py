"""Detached launcher: one job lock, durable exit status, no control-plane credentials."""
import fcntl
from pathlib import Path
import sys
import traceback

from train import main

root=Path(sys.argv[sys.argv.index('--root')+1])
root.mkdir(parents=True,exist_ok=True)
lock=(root/'training.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
code=1
try:
    main()
    code=0
except BaseException:
    traceback.print_exc()
finally:
    (root/'exit_code').write_text(str(code)+'\n')
raise SystemExit(code)
