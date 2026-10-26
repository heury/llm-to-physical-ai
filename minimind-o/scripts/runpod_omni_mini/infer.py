"""Portable inference for the native checkpoint downloaded from Hugging Face."""
import argparse,json,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--model-dir',type=Path,required=True)
 p.add_argument('--prompt',default='Please introduce yourself briefly.')
 p.add_argument('--image',type=Path)
 p.add_argument('--audio',type=Path)
 p.add_argument('--output-wav',type=Path,default=Path('output.wav'))
 p.add_argument('--max-new-tokens',type=int,default=512)
 p.add_argument('--device',default='cuda')
 args=p.parse_args()
 if args.image and args.audio:p.error('Use one input modality per invocation')
 root=args.model_dir.resolve();sys.path.insert(0,str(root/'source'))
 import torch,soundfile as sf
 from PIL import Image,ImageOps
 from huggingface_hub import snapshot_download
 from transformers import AutoTokenizer,MimiModel
 from model.model_omni import MiniMindOmni,OmniConfig
 from dataset.omni_dataset import OmniDataset
 rev=json.loads((root/'revisions.json').read_text())
 def download(repo):return snapshot_download(repo,revision=rev[repo])
 ap=download('jingyaogong/SenseVoiceSmall') if args.audio else '/nonexistent'
 vp=download('google/siglip2-base-patch32-256') if args.image else None
 model=MiniMindOmni(OmniConfig.from_pretrained(root),audio_encoder_path=ap,vision_model_path=vp)
 model.load_state_dict(torch.load(root/'sft_omni_768.pth',map_location='cpu',weights_only=True),strict=True)
 model.eval().to(args.device)
 if model.audio_encoder is not None:model.audio_encoder.to(args.device)
 if model.vision_encoder is not None:model.vision_encoder.to(args.device)
 tok=AutoTokenizer.from_pretrained(root/'tokenizer');prompt=args.prompt;kwargs={}
 if args.audio:
  mel,length=OmniDataset.process_audio(str(args.audio),model.audio_processor)
  kwargs=dict(audio_inputs=mel[None].to(args.device),audio_lens=torch.tensor([length],device=args.device))
  prompt=model.config.audio_special_token*length
 if args.image:
  im=ImageOps.exif_transpose(Image.open(args.image)).convert('RGB')
  kwargs=dict(pixel_values={k:v.to(args.device) for k,v in model.vision_processor(images=im,return_tensors='pt').items()})
  prompt=args.prompt+'\n'+model.config.image_special_token*model.config.image_token_len
 text=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,open_thinking=False)
 ids=torch.tensor([tok.encode(text,add_special_tokens=False)],device=args.device)
 torch.manual_seed(42);frames=[];answer='';steps=0
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
  for y,frame in model.generate(ids,eos_token_id=tok.eos_token_id,max_new_tokens=args.max_new_tokens,temperature=.3,top_p=.9,stream=True,return_audio_codes=True,open_thinking=False,**kwargs):
   steps+=1
   if y is not None:answer=tok.decode(y[0],skip_special_tokens=True)
   if frame is not None:frames.append(frame)
 print(answer)
 if frames:
  mimi=MimiModel.from_pretrained(download('kyutai/mimi')).eval().to(args.device)
  codes=torch.tensor(frames,device=args.device).T[None]
  assert int(codes.min())>=0 and int(codes.max())<2048
  with torch.no_grad():wav=mimi.decode(codes).audio_values.squeeze().float().cpu().numpy()
  args.output_wav.parent.mkdir(parents=True,exist_ok=True);sf.write(args.output_wav,wav,24000)
  print(f'Audio saved: {args.output_wav}')
 print(json.dumps(dict(frames=len(frames),generation_steps=steps,possibly_truncated=steps>=args.max_new_tokens)))
if __name__=='__main__':main()
