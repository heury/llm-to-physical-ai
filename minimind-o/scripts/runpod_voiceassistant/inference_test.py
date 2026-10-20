"""Actual audio-only and image-conditioned inference on the trained checkpoint."""
import os,sys,json,time,hashlib,shutil,tarfile,traceback
from pathlib import Path
import torch,numpy as np,soundfile as sf
from PIL import Image
from transformers import AutoTokenizer,MimiModel
REPO=Path('/tmp/minimind-o-voiceassistant-20261020/repo');sys.path.insert(0,str(REPO))
from model.model_omni import MiniMindOmni,OmniConfig
from dataset.omni_dataset import OmniDataset
from trainer.trainer_utils import setup_seed
ROOT=Path('/shared-volume/minimind-o-voiceassistant-20261020')
OUT=ROOT/'inference-tests';OUT.mkdir(exist_ok=True)
EXPECTED='46a57ec9e305ad6160b1bd73a27d901a29e8b211e2951f56575eab2235c2b7b6'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(n,d):(OUT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2))
def main():
 setup_seed(42);torch.set_num_threads(4)
 weight=ROOT/'out/sft_omni_768.pth';assert sha(weight)==EXPECTED
 model=MiniMindOmni(OmniConfig(hidden_size=768,num_hidden_layers=8),audio_encoder_path=str(ROOT/'models/SenseVoiceSmall'),vision_model_path=str(ROOT/'models/vision'))
 model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
 assert model.audio_encoder is not None and model.vision_encoder is not None,'Both real encoders must load'
 model.eval().cuda();model.audio_encoder.cuda();model.vision_encoder.cuda()
 tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer');mimi=MimiModel.from_pretrained(ROOT/'models/mimi').eval().cuda()
 write('status.json',dict(phase='inference',checkpoint_sha256=EXPECTED,vision_encoder=type(model.vision_encoder).__name__,audio_encoder=type(model.audio_encoder).__name__))
 tests=[('audio-food','audio','audio-en-01_what_do_you_usually_like_to_eat.mp3','What do you usually like to eat?'),('audio-spring','audio','audio-en-04_please_introduce_spring_in_one_sentence.mp3','Please introduce spring in one sentence.'),('image-cat','image','image-01-orange-cat-moon-desk.jpg','Please describe this image.'),('image-dog','image','image-07-golden-retriever-red-scarf-snow.jpg','Please describe this image.')]
 results=json.loads((OUT/'results.json').read_text())['tests'] if (OUT/'results.json').exists() else []
 for name,kind,filename,description in tests:
  if any(t['id']==name for t in results):continue
  setup_seed(42)
  src=REPO/'dataset/eval_omni'/filename;inp=OUT/(name+'-input'+src.suffix);shutil.copyfile(src,inp)
  kwargs={};feature=None
  if kind=='audio':
   mel,length=OmniDataset.process_audio(str(src),model.audio_processor)
   kwargs=dict(audio_inputs=mel[None].cuda(),audio_lens=torch.tensor([length],device='cuda'))
   prompt=model.config.audio_special_token*length
  else:
   im=Image.open(src).convert('RGB');pixels={k:v.cuda() for k,v in model.vision_processor(images=im,return_tensors='pt').items()}
   kwargs=dict(pixel_values=pixels)
   prompt='Please describe this image.\n\n'+model.config.image_special_token*model.config.image_token_len
  messages=[{'role':'system','content':'You are a helpful voice assistant.'},{'role':'user','content':prompt}]
  text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,open_thinking=False)
  ids=torch.tensor([tok.encode(text,add_special_tokens=False)],device='cuda');frames=[];answer='';steps=0;start=time.time()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   if kind=='audio':
    feature=model.encode_audio_inputs(kwargs['audio_inputs'],kwargs['audio_lens'])[0]
   else:feature=model.vision_proj(model.get_image_embeddings(pixels))
   assert feature is not None and torch.isfinite(feature).all() and feature.abs().sum()>0
   # Verify the sensor features change model predictions, compared with zero input.
   actual=model(ids,**kwargs).logits[0,-1].float()
   zero={k:(torch.zeros_like(v) if isinstance(v,torch.Tensor) else {a:torch.zeros_like(b) for a,b in v.items()}) for k,v in kwargs.items()}
   null=model(ids,**zero).logits[0,-1].float()
   delta=(actual-null).abs().max().item();assert delta>1e-5,'Conditioning has no effect'
   for y,fr in model.generate(ids,eos_token_id=tok.eos_token_id,max_new_tokens=256,temperature=.3,top_p=.9,stream=True,return_audio_codes=True,open_thinking=False,**kwargs):
    steps+=1
    if y is not None:answer=tok.decode(y[0],skip_special_tokens=True)
    if fr is not None:frames.append(fr)
  result=dict(id=name,modality=kind,input=inp.name,input_description=description,question_text_provided_to_model=kind!='audio',response_text=answer,frames=len(frames),generation_steps=steps,max_new_tokens=256,possibly_truncated=steps>=256,conditioning_logit_max_abs_difference=delta,feature_shape=list(feature.shape),elapsed_seconds=time.time()-start)
  if frames:
   codes=torch.tensor(frames,device='cuda').T[None];assert codes.min()>=0 and codes.max()<2048
   with torch.no_grad():wav=mimi.decode(codes).audio_values.squeeze().float().cpu().numpy()
   assert np.isfinite(wav).all();path=OUT/(name+'-output.wav');sf.write(path,wav,24000)
   result.update(output_audio=path.name,output_seconds=len(wav)/24000,audio_peak=float(np.abs(wav).max()),audio_rms=float(np.sqrt(np.mean(wav**2))))
  else:result['output_audio']=None
  results.append(result);write('results.json',dict(checkpoint_sha256=EXPECTED,seed=42,tests=results));print(json.dumps(result),flush=True)
 hashes={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='SHA256.json'};write('SHA256.json',hashes)
 with tarfile.open(ROOT/'inference-tests.tar.gz.tmp','w:gz',compresslevel=1) as t:
  for p in OUT.iterdir():t.add(p,arcname=p.name)
 (ROOT/'inference-tests.tar.gz.tmp').replace(ROOT/'inference-tests.tar.gz')
 (OUT/'SUCCESS').write_text('All4realconditionedtestspassed\n')
if __name__=='__main__':
 try:main()
 except BaseException:traceback.print_exc();raise
