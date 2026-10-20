"""Audit the input and synthesized output audio using frozen SenseVoice ASR."""
import json,time,hashlib,tarfile,traceback,re
from pathlib import Path
import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
ROOT=Path('/shared-volume/minimind-o-voiceassistant-20261020');OUT=ROOT/'inference-tests'
for _ in range(180):
 if (OUT/'SUCCESS').exists():break
 time.sleep(5)
else:raise RuntimeError('Inference tests did not complete within15min')
torch.set_num_threads(4)
m=AutoModel(model=str(ROOT/'models/SenseVoiceSmall'),trust_remote_code=True,disable_update=True,device='cpu')
results=json.loads((OUT/'results.json').read_text())
for test in results['tests']:
 for key in ['input','output_audio']:
  if key=='input' and test['modality']!='audio':continue
  if not test.get(key):continue
  try:
   x=m.generate(input=str(OUT/test[key]),cache={},language='en',use_itn=True,batch_size_s=60)
   test[key+'_asr_raw']=x[0]['text'];test[key+'_asr']=rich_transcription_postprocess(x[0]['text'])
  except Exception as e:test[key+'_asr_error']=str(e)
 print(json.dumps(test),flush=True)
(OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
hashes={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['SHA256.json','ASR_COMPLETE']}
(OUT/'SHA256.json').write_text(json.dumps(hashes,indent=2))
with tarfile.open(ROOT/'inference-tests.tar.gz.tmp','w:gz',compresslevel=1) as t:
 for p in OUT.iterdir():t.add(p,arcname=p.name)
(ROOT/'inference-tests.tar.gz.tmp').replace(ROOT/'inference-tests.tar.gz')
(OUT/'ASR_COMPLETE').write_text('complete\n')
