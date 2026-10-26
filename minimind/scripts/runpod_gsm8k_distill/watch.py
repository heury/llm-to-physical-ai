"""Bounded supervisor; stop task GPU on failure/deadline, preserve volume."""
import json,time,subprocess,sys
from remote import OUT,ssh,cli
start=time.time();last=None
while True:
 try:
  r=ssh('if test -f /workspace/moe-dense-gsm8k-distill-20261025/status.json; then cat /workspace/moe-dense-gsm8k-distill-20261025/status.json; fi',capture_output=True,text=True,timeout=45)
  if r.returncode==0 and r.stdout.strip():
   state=json.loads(r.stdout);(OUT/'status.json').write_text(json.dumps(state,indent=2))
   if state!=last:print(json.dumps(state),flush=True);last=state
   if state['phase']=='complete':
    subprocess.run([sys.executable,str(__import__('pathlib').Path(__file__).parent/'finish.py')],check=True);break
   if state['phase']=='failed':raise RuntimeError(state['error'])
  if time.time()-start>4*3600:raise RuntimeError('Four-hour task deadline reached')
 except (subprocess.TimeoutExpired,json.JSONDecodeError):
  print('Transient poll failure',flush=True)
 except Exception as e:
  print('SUPERVISOR_FAILED',repr(e),flush=True)
  pod=json.loads((OUT/'pod.json').read_text())
  try:
   cli('pod','stop',pod['id']);(OUT/'stopped_on_failure.json').write_text(json.dumps(dict(error=repr(e),pod_id=pod['id'],volume_preserved=True)))
  except Exception as stop_error:print('STOP_FAILED',repr(stop_error),flush=True)
  raise
 time.sleep(30)
