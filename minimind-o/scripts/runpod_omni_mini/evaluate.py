"""Generate heldout examples and package a reloadable native checkpoint."""
import os,sys,json,time,shutil,io,zipfile,hashlib,tarfile
from pathlib import Path
ROOT=Path(os.environ.get('OMNI_RUN_ROOT','/workspace/omni-mini-20261025'));REPO=ROOT/'repo'
sys.path.insert(0,str(REPO))
import torch,numpy as np,soundfile as sf,pyarrow.parquet as pq
from PIL import Image,ImageOps
from transformers import AutoTokenizer,MimiModel
from model.model_omni import MiniMindOmni,OmniConfig
from dataset.omni_dataset import OmniDataset
from trainer.trainer_utils import setup_seed
from train import write,sha,phase,ART

def main():
 torch.set_num_threads(4);setup_seed(42)
 assert (ROOT/'TRAINED').exists()
 model=MiniMindOmni(OmniConfig(hidden_size=768,num_hidden_layers=8),audio_encoder_path=str(ROOT/'models/audio'),vision_model_path=str(ROOT/'models/vision'))
 model.load_state_dict(torch.load(ART/'sft_omni_768.pth',map_location='cpu',weights_only=True),strict=True)
 model.eval().cuda();model.audio_encoder.cuda().eval();model.vision_encoder.cuda().eval()
 tok=AutoTokenizer.from_pretrained(ART/'tokenizer');mimi=MimiModel.from_pretrained(ROOT/'models/mimi').eval().cuda()
 samples=ART/'samples';samples.mkdir(exist_ok=True);tests=[]
 splits=json.loads((ART/'audio_splits.json').read_text())
 for kind in ['t2a','a2a']:
  table=pq.read_table(ROOT/f'data/audio/sft_{kind}_mini.parquet')
  for j,idx in enumerate(splits[kind]['validation'][:2]):
   row=table.slice(idx,1).to_pylist()[0];convs=json.loads(row['conversations'])
   # First complete turn avoids accidentally passing a reference assistant answer.
   user=next(t['content'] for t in convs if t['role']=='user')
   expected=next(t['content'] for t in convs if t['role']=='assistant')
   name=f'{kind}-{j+1}';kwargs={};prompt=user;input_file=None
   if kind=='a2a':
    raw=row['question_audios'][0];wave,sr=sf.read(io.BytesIO(raw));input_file=f'{name}-input.wav';sf.write(samples/input_file,wave,sr)
    mel,length=OmniDataset.process_audio(str(samples/input_file),model.audio_processor)
    kwargs=dict(audio_inputs=mel[None].cuda(),audio_lens=torch.tensor([length],device='cuda'))
    prompt=model.config.audio_special_token*length
   tests.append(dict(id=name,kind=kind,prompt=prompt,kwargs=kwargs,source_row=idx,input_text=user,reference_text=expected,input_file=input_file))
  del table
 rows=pq.read_table(ROOT/'data/vision/conversations/sft_test/00000.parquet').to_pylist()
 selected=[];seen_images=set()
 for row in rows:
  if row['pixel_sha256'] not in seen_images:
   selected.append(row);seen_images.add(row['pixel_sha256'])
  if len(selected)==3:break
 for j,row in enumerate(selected):
  with zipfile.ZipFile(ROOT/'data/vision'/row['image_archive']) as z:raw=z.read(row['image'])
  im=ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB');name=f'image-{j+1}';input_file=f'{name}-input.jpg';im.save(samples/input_file)
  user=row['conversations'][0]['value'];expected=row['conversations'][1]['value']
  pixels={k:v.cuda() for k,v in model.vision_processor(images=im,return_tensors='pt').items()}
  prompt=user.replace('<image>',model.config.image_special_token*model.config.image_token_len)
  tests.append(dict(id=name,kind='image',prompt=prompt,kwargs=dict(pixel_values=pixels),source_id=row['id'],input_text=user,reference_text=expected,input_file=input_file))
 results=[]
 for test in tests:
  phase('generating',sample=test['id']);setup_seed(42)
  prompt=tok.apply_chat_template([{'role':'user','content':test['prompt']}],tokenize=False,add_generation_prompt=True,open_thinking=False)
  ids=torch.tensor([tok.encode(prompt,add_special_tokens=False)],device='cuda')
  frames=[];answer='';steps=0;start=time.time()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   delta=None
   if test['kwargs']:
    actual=model(ids,**test['kwargs']).logits[0,-1].float()
    zero={k:({a:torch.zeros_like(b) for a,b in v.items()} if isinstance(v,dict) else torch.zeros_like(v)) for k,v in test['kwargs'].items()}
    null=model(ids,**zero).logits[0,-1].float();delta=float((actual-null).abs().max());assert delta>1e-5
   for y,frame in model.generate(ids,eos_token_id=tok.eos_token_id,max_new_tokens=512,temperature=.3,top_p=.9,stream=True,return_audio_codes=True,open_thinking=False,**test['kwargs']):
    steps+=1
    if y is not None:answer=tok.decode(y[0],skip_special_tokens=True)
    if frame is not None:frames.append(frame)
  result={k:v for k,v in test.items() if k not in ['prompt','kwargs']}
  result.update(response_text=answer,frames=len(frames),generation_steps=steps,max_new_tokens=512,possibly_truncated=steps>=512,conditioning_logit_difference=delta,elapsed_seconds=time.time()-start)
  if frames:
   codes=torch.tensor(frames,device='cuda').T[None];assert codes.min()>=0 and codes.max()<2048
   with torch.no_grad():wav=mimi.decode(codes).audio_values.squeeze().float().cpu().numpy()
   assert np.isfinite(wav).all();path=samples/f"{test['id']}-output.wav";sf.write(path,wav,24000)
   result.update(output_audio=path.name,seconds=len(wav)/24000,rms=float(np.sqrt(np.mean(wav**2))))
  else:result['output_audio']=None
  results.append(result);write(samples/'results.json',results);print(json.dumps(result,ensure_ascii=False),flush=True)
 for kind in ['t2a','a2a']:
  assert any(r['kind']==kind and r.get('output_audio') and r.get('rms',0)>1e-6 for r in results),f'No usable generated audio for {kind}'
 assert all(r['response_text'].strip() for r in results),'Empty generated text'
 # Automatic ASR is an auxiliary transcription check, not human listening.
 phase('sample_asr')
 from funasr import AutoModel
 from funasr.utils.postprocess_utils import rich_transcription_postprocess
 asr=AutoModel(model=str(ROOT/'models/audio'),trust_remote_code=True,disable_update=True,device='cpu')
 for result in results:
  if result.get('output_audio'):
   try:
    transcription=asr.generate(input=str(samples/result['output_audio']),cache={},language='en',use_itn=True,batch_size_s=60)
    result['output_audio_asr']=rich_transcription_postprocess(transcription[0]['text'])
   except Exception as e:result['output_audio_asr_error']=str(e)
 write(samples/'results.json',results)
 # Package only deliverable weights, exact source, metrics and examples, never tokens/caches.
 for folder in ['model','dataset','trainer']:
  dest=ART/'source'/folder;dest.mkdir(parents=True,exist_ok=True)
  for p in (REPO/folder).glob('*.py'):shutil.copyfile(p,dest/p.name)
 shutil.copytree(REPO/'scripts/runpod_omni_mini',ART/'source/scripts/runpod_omni_mini',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 import subprocess
 (ART/'requirements-resolved.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 (ART/'README.md').write_text('''---
language:
- en
license: other
license_name: upstream-model-and-dataset-terms
license_link: https://huggingface.co/datasets/caspar/MiniMind-V-LLaVA-Short-Training-v1.0/blob/main/LICENSE_NOTES.md
base_model: jingyaogong/minimind-3o-pytorch
datasets:
- jingyaogong/minimind-o_dataset
- caspar/MiniMind-V-LLaVA-Short-Training-v1.0
pipeline_tag: any-to-any
---
# MiniMind-O Mini + LLaVA — RTX 5090 fine-tuning

Native dense MiniMind-O checkpoint fine-tuned from published `sft_omni_768.pth`.
All training rows of `sft_t2a_mini` and `sft_a2a_mini`, excluding 128 heldout rows each;
all provided visual caption and SFT training rows; original visual heldouts preserved.
One epoch each: T2A full voice model, A2A projector alignment, A2A full voice model,
visual caption projector alignment, visual QA projector tuning. Frozen SenseVoice,
SigLIP2 and Mimi; visual stages update only vision_proj to preserve voice parameters.

See `run_config.json`, `training_summary.json`, `metrics/`, `training.jsonl` and
`samples/results.json` for exact settings, measured losses, samples and limitations.
Audio losses use 10x stop-token weights. Validation is teacher-forced; no claim of
human-rated speech quality. Base-pretraining overlap with heldouts is unknown.
Generated examples may be inaccurate or truncated; inspect their recorded flags.

## Load the native model

Install `requirements-resolved.txt` in a CUDA-compatible environment, and put
`source/` on Python's import path. Download frozen encoder repositories at revisions
listed in `revisions.json`: SenseVoiceSmall, google/siglip2-base-patch32-256, kyutai/mimi.

```python
import sys, torch
sys.path.insert(0, "source")
from model.model_omni import MiniMindOmni, OmniConfig
from transformers import AutoTokenizer
model = MiniMindOmni(OmniConfig(hidden_size=768, num_hidden_layers=8),
    audio_encoder_path="models/audio", vision_model_path="models/vision")
model.load_state_dict(torch.load("sft_omni_768.pth", map_location="cpu", weights_only=True), strict=True)
model.eval().cuda()
if model.audio_encoder is not None: model.audio_encoder.cuda()
if model.vision_encoder is not None: model.vision_encoder.cuda()
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
```

Portable inference (from the downloaded snapshot directory):

```bash
python source/scripts/runpod_omni_mini/infer.py --model-dir . --prompt "Please introduce yourself briefly." --output-wav reply.wav
python source/scripts/runpod_omni_mini/infer.py --model-dir . --image photo.jpg --prompt "Describe the image."
python source/scripts/runpod_omni_mini/infer.py --model-dir . --audio question.wav
```

The script downloads pinned frozen models when needed. It defaults to CUDA.
This is a native PyTorch state dict, not an AutoModel export. See the included
`evaluate.py` for text/audio/image conditioning and Mimi waveform decoding.
External frozen models are downloaded separately, not included in this repository.

## Source terms

MiniMind-O source code is Apache-2.0; its license is included under `provenance/MINIMIND_O_CODE_LICENSE`. Preserve upstream model terms and dataset/image attribution. LLaVA
pretrain images are research-reproduction data; no blanket commercial image license
is granted. Dataset source cards and license notes are retained under `provenance/`.
''')
 prov=ART/'provenance';prov.mkdir(exist_ok=True)
 shutil.copyfile(REPO/'LICENSE',prov/'MINIMIND_O_CODE_LICENSE')
 shutil.copyfile(ROOT/'data/vision/LICENSE_NOTES.md',prov/'LICENSE_NOTES.md')
 shutil.copytree(ROOT/'data/vision/provenance/source-cards',prov/'source-cards',dirs_exist_ok=True)
 summary=json.loads((ART/'training_summary.json').read_text())
 lines=['# Omni 미세조정 실행 결과', '', 'RTX 5090 1개에서 공개 Omni 가중치를 미세조정했습니다. 정확한 설정과 원본 버전은 run_config.json에 있습니다.', '', '| 단계 | 학습 행 | 업데이트 | 시간(분) |', '|---|---:|---:|---:|']
 for stage in summary['stages']:
  lines.append(f"| {stage['stage']} | {stage['processed']:,} | {stage['optimizer_updates']:,} | {stage['elapsed_seconds']/60:.1f} |")
 lines += ['', '## 검증 손실', '', '오디오 손실에는 종료 토큰 10배 가중치가 적용됩니다. 생성 음성의 청취 품질 점수가 아닙니다.', '', '| 평가 | 텍스트 손실 | 오디오 손실 | 샘플 수 |', '|---|---:|---:|---:|']
 for p in sorted((ART/'metrics').glob('*.json')):
  m=json.loads(p.read_text())
  if 'tag' in m:lines.append(f"| {m['tag']} | {m['text_loss']:.4f} | {m['audio_weighted_loss']:.4f} | {m['examples']} |")
 lines += ['', '## 생성 샘플', '']
 for r in results:
  lines += [f"### {r['id']}", '', f"입력: {r['input_text']}", '', f"참고 정답: {r['reference_text']}", '', f"모델 응답: {r['response_text']}", '', f"생성 프레임: {r['frames']}; 길이 제한 도달: {r['possibly_truncated']}", '']
  if r.get('output_audio_asr'):lines += [f"생성 음성 자동 전사(SenseVoice): {r['output_audio_asr']}", '']
  if r.get('input_file'):lines += [f"[입력 파일](samples/{r['input_file']})", '']
  if r.get('output_audio'):lines += [f"[생성 음성](samples/{r['output_audio']})", '']
 lines += ['## 검증 범위', '', '음성 검증은 행 단위로 분리했고 이미지 데이터의 기존 분할을 유지했습니다. 초기 공개 모델의 사전학습 데이터와 검증 샘플의 중복 여부는 확인할 수 없습니다. 생성 응답의 사실성과 발음 품질은 자동 손실만으로 보장하지 않습니다.']
 (ART/'REPORT_ko.md').write_text('\n'.join(lines)+'\n')
 smoke=subprocess.run([sys.executable,str(ART/'source/scripts/runpod_omni_mini/infer.py'),'--model-dir',str(ART),'--prompt','Hello!','--max-new-tokens','2','--device','cpu'],capture_output=True,text=True,timeout=120,check=True)
 (ART/'portable_inference_smoke.txt').write_text(smoke.stdout)
 hashes={str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
 write(ART/'SHA256SUMS.json',hashes)
 with tarfile.open(ROOT/'artifacts.tar.gz.tmp','w:gz',compresslevel=1) as tar:tar.add(ART,arcname='artifacts')
 (ROOT/'artifacts.tar.gz.tmp').replace(ROOT/'artifacts.tar.gz')
 write(ROOT/'SUCCESS.json',dict(bundle_sha256=sha(ROOT/'artifacts.tar.gz'),checkpoint_sha256=sha(ART/'sft_omni_768.pth'),samples=len(results)))
 phase('complete',samples=len(results))
if __name__=='__main__':main()
