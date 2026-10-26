"""Build, verify and publicly publish only the final inference files."""
import importlib.util,json,shutil,sys,time
from pathlib import Path
import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi,snapshot_download
from common import sha,write
REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'outputs/minimind-v/llava-20261025'
BUNDLE=OUT/'bundle'
NAME='MiniMind-V-Dense-64M-LLaVA-558K-150K-v1.0'
ID='caspar/'+NAME

def main():
    if (OUT/'published.json').exists():return
    done=json.loads((BUNDLE/'completion.json').read_text());report=done['report']
    assert done['complete'] and report['finite_weights'] and report['strict_nonvision_reload']
    assert sha(BUNDLE/'sft_vlm_768.pth')==report['weights_sha256']
    folder=OUT/NAME;folder.mkdir(exist_ok=True)
    for name in ['vision_encoder']:
        shutil.copytree(BUNDLE/name,folder/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.cache'))
    shutil.copytree(BUNDLE/'model',folder/'minimind_v',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','model_lora.py'))
    (folder/'minimind_v/__init__.py').write_text('')
    for name in ['config.json']:
        shutil.copyfile(BUNDLE/name,folder/name)
    config=json.loads((folder/'config.json').read_text());config['model_version']='v1.0';write(folder/'config.json',config)
    for p in (BUNDLE/'tokenizer').iterdir():
        if p.is_file():shutil.copyfile(p,folder/p.name)
    shutil.copyfile(Path(__file__).parent/'inference.py',folder/'inference.py')
    state=torch.load(BUNDLE/'sft_vlm_768.pth',map_location='cpu',weights_only=True)
    assert torch.equal(state['lm_head.weight'],state['model.embed_tokens.weight'])
    del state['lm_head.weight']
    save_file({k:v.contiguous().clone() for k,v in state.items()},folder/'model.safetensors',metadata={'format':'pt'})
    spec=importlib.util.spec_from_file_location('release_inference',folder/'inference.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(4);model,tok=module.load_model(folder)
    for k,v in state.items():assert torch.equal(v,model.state_dict()[k].half()),k
    from PIL import Image
    im=Image.open(BUNDLE/'samples/00.jpg')
    response=module.answer(model,tok,im,'Describe this image.',max_new_tokens=16)
    assert response.strip()
    write(OUT/'release_smoke.json',dict(weights_equal=True,strict_nonvision_load=True,finite_weights=True,generation_smoke=response))
    shutil.copyfile(REPO/'minimind/LICENSE',folder/'LICENSE-CODE')
    (folder/'requirements.txt').write_text('torch>=2.6,<3\ntransformers==4.57.6\nsafetensors>=0.4\nPillow>=10\nhuggingface_hub>=0.36\n')
    write(folder/'evaluation.json',report)
    manifest=json.loads((BUNDLE/'dataset_manifest.json').read_text())
    core=sum(p.numel() for n,p in model.named_parameters() if not n.startswith('vision_encoder.'))
    total=sum(p.numel() for p in model.parameters())
    card=f'''---
license: other
license_name: llava-research
license_link: https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
language:
- en
pipeline_tag: image-text-to-text
tags:
- minimind-v
- siglip2
- llava
- experimental
datasets:
- liuhaotian/LLaVA-Pretrain
- liuhaotian/LLaVA-Instruct-150K
base_model:
- caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0
- google/siglip2-base-patch32-256
---

# {NAME}

Version **v1.0**. MiniMind-V with a 64M Dense language backbone, SigLIP2 base patch32-256 frozen vision encoder and MLP projector.
Actual parameters: {core:,} language/projector; {total:,} including frozen vision encoder.
The source language model is `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`.

## Training

Original LLaVA-Pretrain `blip_laion_cc_sbu_558k.json` (LAION/CC/SBU mixture; not CC3M alone), followed by original LLaVA-Instruct-150K `llava_instruct_150k.json` and COCO 2017 train images.
One epoch projector-only alignment (LR 4e-4); one epoch projector plus first/last language-layer SFT (LR 1e-5). Vision encoder frozen in both phases.
BF16; batch 64, accumulation 2; 768 tokens and 64 visual tokens. Full assistant targets/EOS retained by overlapping windows with repeated image prefix; overlapping labels masked. Later windows may not contain full earlier conversation context.
Original row counts: pretrain {manifest['pretrain']['source_rows']:,}, SFT {manifest['sft']['source_rows']:,}. See evaluation.json and training_manifest.json for retained rows, windows and target counts.
Splits are grouped by decoded 256x256 RGB pixel hash, with SFT validation/test images excluded from alignment. This detects exact resized-pixel duplicates, not all near duplicates or prior language pretraining leakage.

## Evaluation

Held-out SFT test PPL: {report['test']['perplexity']:.4f}.
Same text with black-image input PPL: {report['test_black_image']['perplexity']:.4f}.
These are teacher-forced likelihood diagnostics, not a general visual QA accuracy score. Eight held-out image samples were generated using greedy, sampling and black-image controls. No test answers were used for training or checkpoint selection.
The base language model has weak factual/math reasoning; hallucinations, repetition, incorrect counting and instruction-following failures remain possible. Do not interpret fluency as grounded correctness. Evaluated training context is 768 tokens.

## Files and usage

This is a native MiniMind-V inference bundle. Use the included loader, which loads the separate vision encoder before strictly checking language/projector weights. No AutoModel remote-code support is claimed.

```bash
pip install -r requirements.txt
python inference.py --model . --image your_image.jpg --question "What is in this image?"
```

Or download versioned files:

```python
from huggingface_hub import snapshot_download
path = snapshot_download("{ID}", revision="v1.0")
# Run: python <path>/inference.py --model <path> --image your_image.jpg
```

## Sources and usage terms

- MiniMind code: https://github.com/jingyaogong/minimind (Apache-2.0; LICENSE-CODE).
- SigLIP2: https://huggingface.co/google/siglip2-base-patch32-256 (Apache-2.0).
- https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
- https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K
- COCO images retain their original licenses; see https://cocodataset.org/.

Research release. Dataset-generated instruction terms and source image licenses apply; this release does not grant new rights to source data or assert unrestricted commercial use. Raw datasets, optimizer states, credentials and W&B logs are not included.
'''
    (folder/'README.md').write_text(card)
    write(folder/'training_manifest.json',dict(dataset=manifest,settings=json.loads((BUNDLE/'settings.json').read_text())))
    write(folder/'release.json',dict(model_name=NAME,version='v1.0',source_checkpoint_sha256=report['weights_sha256'],files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='release.json' and '__pycache__' not in p.parts}))
    for p in folder.rglob('__pycache__'):shutil.rmtree(p)
    api=HfApi();assert api.whoami()['name']=='caspar'
    if not (OUT/'repo_created.json').exists():
        api.create_repo(ID,private=False,exist_ok=False);write(OUT/'repo_created.json',dict(repo_id=ID))
    commit=api.upload_folder(repo_id=ID,folder_path=folder,commit_message='v1.0: final LLaVA alignment + instruction SFT')
    api.create_tag(ID,tag='v1.0',revision=commit.oid,exist_ok=True)
    info=HfApi(token=False).model_info(ID,revision='v1.0');assert not info.private and info.sha==commit.oid
    downloaded=Path(snapshot_download(ID,revision=commit.oid,token=False,local_dir=OUT/'public_verification'))
    for p in folder.rglob('*'):
        if p.is_file():assert sha(p)==sha(downloaded/p.relative_to(folder))
    write(OUT/'published.json',dict(repo_id=ID,url='https://huggingface.co/'+ID,version='v1.0',commit=commit.oid,public=True,anonymous_hashes_verified=True,timestamp=time.time()))

if __name__=='__main__':main()
