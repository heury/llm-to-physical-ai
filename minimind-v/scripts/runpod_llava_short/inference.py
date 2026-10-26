"""Load the released MiniMind-V bundle and answer a question about one image."""
from pathlib import Path
import sys
import torch
from PIL import Image, ImageOps
from transformers import AutoTokenizer
from safetensors.torch import load_file

def load_model(directory, device='cpu'):
    directory=Path(directory).resolve();sys.path.insert(0,str(directory))
    from minimind_v.model_vlm import MiniMindVLM,VLMConfig
    config=VLMConfig.from_pretrained(directory,local_files_only=True)
    model=MiniMindVLM(config,vision_model_path=str(directory/'vision_encoder'))
    assert model.vision_encoder is not None
    state=load_file(directory/'model.safetensors')
    if config.tie_word_embeddings:state['lm_head.weight']=state['model.embed_tokens.weight']
    missing=model.load_state_dict(state,strict=False)
    expected={k for k in model.state_dict() if k.startswith('vision_encoder.')}
    assert not missing.unexpected_keys and set(missing.missing_keys)==expected
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
    assert tokenizer.convert_tokens_to_ids('<|image_pad|>')==config.image_ids[0]==12
    return model.to(device).eval(),tokenizer

@torch.inference_mode()
def answer(model,tokenizer,image,question,max_new_tokens=128):
    image=ImageOps.exif_transpose(image).convert('RGB')
    text=tokenizer.apply_chat_template([{'role':'user','content':'<|image_pad|>'*64+'\n'+question}],tokenize=False,add_generation_prompt=True,open_thinking=False)
    text=text.replace('<think>\n\n</think>\n\n','')
    device=next(model.parameters()).device
    x=tokenizer(text,return_tensors='pt',add_special_tokens=False).input_ids.to(device)
    budget=min(max_new_tokens,340-x.shape[1]);assert budget>0
    pixels={k:v.to(device) for k,v in model.processor(images=image,return_tensors='pt').items()}
    y=model.generate(x,pixel_values=pixels,max_new_tokens=budget,do_sample=False,temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,eos_token_id=tokenizer.eos_token_id,use_cache=True)
    return tokenizer.decode(y[0,x.shape[1]:],skip_special_tokens=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--model',default=str(Path(__file__).parent));p.add_argument('--image',required=True);p.add_argument('--question',default='Describe this image.');p.add_argument('--device',default='cpu')
    args=p.parse_args();torch.set_num_threads(4)
    model,tokenizer=load_model(args.model,args.device)
    with Image.open(args.image) as im:print(answer(model,tokenizer,im,args.question))
