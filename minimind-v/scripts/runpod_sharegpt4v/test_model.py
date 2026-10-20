"""Read-only GPU test of the final MiniMind-V bundle, with image controls."""
import hashlib,json,math,os,random,shutil,subprocess,sys,tarfile,time,traceback
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import torch
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM,VLMConfig

BASE=Path('/shared-volume/minimind-v-sharegpt4v-20261020')
RUNTIME=Path('/tmp/minimind-v-test-20261020')
OUT=BASE/'tests/20261020-runpod'
EXPECTED='4e40a7e96bd1e846f9f2c0a5ed29eecf6e7701480f0eb4a33f24aa603590df58'
CAPTION='Describe the main objects and scene in this image.'
CASES=[
 ('image-01-golden-dog-balloons.jpg','What animal is in the image?','A golden dog is running on grass with colorful balloons.'),
 ('image-02-rainbow-umbrella-street.jpg','What object is in the foreground?','A rainbow umbrella is on a wet city street.'),
 ('image-03-cherry-blossom-bike.jpg','What is the main vehicle?','A blue bicycle with a flower basket stands under pink blossoms.'),
 ('image-04-yellow-car.jpg','What color is the car?','A yellow sports car is parked on a road.'),
 ('image-05-superhero-rooftop.jpg',"What color is the person's cape?",'A superhero with a red cape stands on a rooftop at sunset.'),
 ('image-06-racecar-drift.jpg','What color is the race car?','A red race car is driving on a racetrack.'),
]

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,data):
 p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2));t.replace(p)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 assert not (OUT/'SUCCESS').exists(),'Do not overwrite a completed test'
 torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);random.seed(42)
 assert torch.cuda.is_available(),'GPU required'
 device='cuda';torch.cuda.set_per_process_memory_fraction(.5)
 weight=BASE/'out/sft_vlm_768.pth';assert sha(weight)==EXPECTED
 config=VLMConfig.from_pretrained(BASE,local_files_only=True)
 model=MiniMindVLM(config,vision_model_path=str(BASE/'vision_encoder'))
 assert model.vision_encoder is not None and model.processor is not None
 state=torch.load(weight,map_location='cpu',weights_only=True)
 assert all(torch.isfinite(x).all() for x in state.values())
 result=model.load_state_dict(state,strict=False)
 expected_missing={k for k in model.state_dict() if k.startswith('vision_encoder.')}
 assert not result.unexpected_keys and set(result.missing_keys)==expected_missing
 assert all(torch.isfinite(p).all() for p in model.vision_encoder.parameters())
 tokenizer=AutoTokenizer.from_pretrained(BASE/'tokenizer',local_files_only=True)
 assert tokenizer.convert_tokens_to_ids('<|image_pad|>')==config.image_ids[0]==12
 model=model.to(device).eval();processor=model.processor
 for parameter in model.parameters():parameter.requires_grad=False
 pixels=[]
 for filename,_,_ in CASES:
  path=RUNTIME/'images'/filename
  with Image.open(path) as im:
   im=ImageOps.exif_transpose(im).convert('RGB').resize((256,256),Image.Resampling.BICUBIC)
   pixels.append(processor(images=im,return_tensors='pt')['pixel_values'].to(device))
 black=processor(images=Image.new('RGB',(256,256),'black'),return_tensors='pt')['pixel_values'].to(device)
 def messages(question,answer=None):
  rows=[dict(role='system',content='You are a helpful assistant.'),dict(role='user',content='<|image_pad|>'*64+'\n'+question)]
  if answer is not None:rows.append(dict(role='assistant',content=answer))
  return rows
 def prompt(question):
  ids=tokenizer.apply_chat_template(messages(question),tokenize=True,add_generation_prompt=True,open_thinking=False)
  assert ids.count(12)==64
  return torch.tensor([ids],device=device)
 @torch.inference_mode()
 def generate(question,pixel,mode='greedy',limit=128):
  ids=prompt(question)
  opts=dict(max_new_tokens=limit,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
  if mode=='sampled':opts.update(do_sample=True,temperature=.7,top_p=.9,top_k=0,repetition_penalty=1.15,no_repeat_ngram_size=3)
  torch.manual_seed(42);torch.cuda.synchronize();start=time.perf_counter()
  with torch.autocast('cuda',dtype=torch.bfloat16):out=model.generate(ids,attention_mask=torch.ones_like(ids),pixel_values={'pixel_values':pixel},**opts)
  torch.cuda.synchronize();elapsed=time.perf_counter()-start
  new=out[0,ids.shape[1]:].tolist();tokens=[x for x in new if x not in [tokenizer.eos_token_id,tokenizer.pad_token_id]]
  grams=[tuple(tokens[i:i+4]) for i in range(max(0,len(tokens)-3))]
  repeated=1-len(set(grams))/len(grams) if grams else 0.
  return dict(question=question,decoding=opts,answer=tokenizer.decode(new,skip_special_tokens=True),generated_tokens=len(new),seconds=elapsed,tokens_per_second=len(new)/elapsed,eos=tokenizer.eos_token_id in new,repeated_token_4gram_fraction=repeated)
 @torch.inference_mode()
 def last_logits(pixel):
  ids=prompt(CAPTION)
  with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(ids,attention_mask=torch.ones_like(ids),pixel_values={'pixel_values':pixel}).logits[0,-1].float()
  assert torch.isfinite(logits).all();return logits
 # One unrecorded forward warms the GPU before timing actual generation.
 black_logits=last_logits(black);pblack=black_logits.softmax(-1)
 generated=[];controls=[]
 for i,(filename,question,truth) in enumerate(CASES):
  entry=dict(image=filename,observed_image_description=truth,
   caption_greedy=generate(CAPTION,pixels[i]),
   caption_sampled=generate(CAPTION,pixels[i],mode='sampled'),
   focused_question=generate(question,pixels[i],limit=48))
  if i in [0,1,3]:entry['korean']=generate('이 이미지에서 보이는 주요 물체와 장면을 짧게 설명해 주세요.',pixels[i])
  entry['black_image_control']=generate(CAPTION,black)
  logits=last_logits(pixels[i]);p=logits.softmax(-1);m=(p+pblack)/2
  js=.5*((p*(p.log()-m.log())).sum()+(pblack*(pblack.log()-m.log())).sum())
  controls.append(dict(image=filename,logit_max_abs_difference_from_black=(logits-black_logits).abs().max().item(),next_token_total_variation=(p-pblack).abs().sum().item()/2,next_token_js_divergence=js.item()))
  generated.append(entry);write('generations.json',generated)
  print(json.dumps(dict(image=filename,caption=entry['caption_greedy']['answer'],focused=entry['focused_question']['answer']),ensure_ascii=False),flush=True)
 # A small diagnostic, not an independent benchmark: short visually inspected
 # candidate captions, tested against every image using token-mean NLL.
 matrix=[]
 with torch.inference_mode():
  for pixel in pixels:
   row=[]
   for _,_,answer in CASES:
    prefix=tokenizer.apply_chat_template(messages(CAPTION),tokenize=True,add_generation_prompt=True,open_thinking=False)
    ids=tokenizer.apply_chat_template(messages(CAPTION,answer),tokenize=True,add_generation_prompt=False,open_thinking=False)
    assert ids[:len(prefix)]==prefix
    labels=[-100]*len(prefix)+ids[len(prefix):]
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(torch.tensor([ids],device=device),labels=torch.tensor([labels],device=device),pixel_values={'pixel_values':pixel}).loss
    assert torch.isfinite(loss);row.append(loss.item())
   matrix.append(row)
 write('image_conditioning.json',dict(first_token_controls=controls,candidate_caption_nll=matrix,caption_order=[x[2] for x in CASES],note='Six hand-inspected repository illustration images; exploratory diagnostic, not a benchmark. Candidate token lengths differ.'))
 assert sha(weight)==EXPECTED,'Test modified checkpoint'
 speeds=[entry['caption_greedy']['tokens_per_second'] for entry in generated]
 summary=dict(checkpoint_sha256=EXPECTED,finite_checkpoint=True,expected_nonvision_keys_loaded=True,vision_encoder_loaded=True,checkpoint_unchanged=True,
   device=torch.cuda.get_device_name(),images=len(CASES),generation_cases=sum(4+('korean' in x) for x in generated),greedy_caption_mean_tokens_per_second=sum(speeds)/len(speeds),
   mean_repeated_4gram_fraction=float(np.mean([x['caption_greedy']['repeated_token_4gram_fraction'] for x in generated])),
   greedy_caption_eos_count=sum(x['caption_greedy']['eos'] for x in generated),
   conditioning_nll_correct_mean=float(np.mean(np.diag(matrix))),conditioning_nll_wrong_mean=float(np.mean([matrix[i][j] for i in range(6) for j in range(6) if i!=j])),
   peak_cuda_bytes=torch.cuda.max_memory_allocated(),image_files=[x[0] for x in CASES])
 write('report.json',summary)
 finalize(summary,generated)

def finalize(summary,generated):
 (OUT/'images').mkdir(exist_ok=True)
 for image in (RUNTIME/'images').glob('*.jpg'):shutil.copyfile(image,OUT/'images'/image.name)
 shutil.copyfile(__file__,OUT/'test_model.py')
 shutil.copyfile(BASE/'config.json',OUT/'model_config.json')
 subprocess.run([sys.executable,'-m','pip','freeze'],stdout=open(OUT/'requirements-resolved.txt','w'),check=True)
 lines=['MiniMind-V Runpod inference test','',json.dumps(summary,ensure_ascii=False,indent=2),'']
 for x in generated:
  lines.extend([x['image'],'Observed: '+x['observed_image_description']])
  for key in ['caption_greedy','caption_sampled','focused_question','korean','black_image_control']:
   if key in x:lines.extend([key+': '+x[key]['question'],x[key]['answer'],''])
 (OUT/'results.txt').write_text('\n'.join(lines))
 files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['SHA256.json','artifacts.tar.gz','test.log','exit_code','SUCCESS']]
 hashes={str(p.relative_to(OUT)):sha(p) for p in files};write('SHA256.json',hashes)
 with tarfile.open(OUT/'artifacts.tar.gz','w:gz') as tar:
  for name in hashes:tar.add(OUT/name,arcname=name)
  tar.add(OUT/'SHA256.json',arcname='SHA256.json')
 (OUT/'SUCCESS').write_text('completed\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':
 code=1
 try:main();code=0
 except BaseException:traceback.print_exc()
 finally:
  OUT.mkdir(parents=True,exist_ok=True);(OUT/'exit_code').write_text(str(code)+'\n')
 raise SystemExit(code)
