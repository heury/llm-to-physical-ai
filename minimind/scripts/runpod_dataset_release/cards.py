"""Attach self-contained provenance, recipes and dataset cards to exported data."""
import json, shutil
from pathlib import Path
BASE=Path(__file__).resolve().parents[3]
RELEASE=BASE/'outputs/dataset-release-v1.0'
REPOS={'pretrain':'caspar/MiniMind-FineWeb-Edu-Score35-v1.0','sft':'caspar/MiniMind-Mixed-SFT-Training-v1.0','vision':'caspar/MiniMind-V-LLaVA-Short-Training-v1.0'}
SOURCES={
'fineweb':('HuggingFaceFW/fineweb-edu','87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'),
'ultrachat':('HuggingFaceH4/ultrachat_200k','8049631c405ae6576f93f445c6b8166f76f5505a'),
'smoltalk':('HuggingFaceTB/smoltalk','5feaf2fd3ffca7c237fc38d1861bc30365d48ffa'),
'gsm8k':('openai/gsm8k','740312add88f781978c0658806c59bc2815b9866'),
'llava_pretrain':('liuhaotian/LLaVA-Pretrain','70f9d1e5e1a697fe35830875cfc7de1dd590d727'),
'llava_instruct':('liuhaotian/LLaVA-Instruct-150K','9d451dc7629cfe0469f6ae4432b765cd603d5fcb')}

def attach(kind,dest):
    count=json.loads((dest/'export-counts.json').read_text())
    source_names={'pretrain':['fineweb'],'sft':['ultrachat','smoltalk','gsm8k'],'vision':['llava_pretrain','llava_instruct']}[kind]
    (dest/'provenance/source-cards').mkdir(parents=True,exist_ok=True)
    for n in source_names:shutil.copyfile(RELEASE/'source-cards'/f'{n}.md',dest/'provenance/source-cards'/f'{n}.md')
    (dest/'provenance/sources.json').write_text(json.dumps({n:dict(repo=SOURCES[n][0],revision=SOURCES[n][1]) for n in source_names},indent=2))
    code={'pretrain':BASE/'minimind/scripts/runpod_full_pretrain','sft':BASE/'minimind/scripts/runpod_mixed_sft','vision':BASE/'minimind-v/scripts/runpod_llava_short'}[kind]
    for name in ['prepare.py','common.py','filter_data.py']:
        if (code/name).exists():
            (dest/'recipe').mkdir(exist_ok=True);shutil.copyfile(code/name,dest/'recipe'/name)
    shutil.copyfile(Path(__file__).with_name('export.py'),dest/'recipe/export.py')
    if kind=='pretrain':shutil.copyfile(Path(__file__).with_name('fineweb_attribution.py'),dest/'recipe/fineweb_attribution.py')
    if kind=='vision':shutil.copyfile(Path(__file__).with_name('vision_attribution.py'),dest/'recipe/vision_attribution.py')
    if kind=='sft':shutil.copyfile(BASE/'minimind/dataset/lm_dataset.py',dest/'recipe/lm_dataset.py')
    elif kind=='vision':shutil.copyfile(BASE/'minimind-v/dataset/lm_dataset.py',dest/'recipe/lm_dataset.py')
    header='---\nlanguage:\n- en\n'
    if kind=='pretrain':
        header+='license: odc-by\nconfigs:\n- config_name: documents\n  default: true\n  data_files:\n  - split: train\n    path: documents/train/*.parquet\n  - split: validation\n    path: documents/validation/*.parquet\n- config_name: excluded_documents\n  data_files:\n  - split: excluded\n    path: documents/excluded/*.parquet\n- config_name: attribution\n  data_files:\n  - split: source\n    path: attribution/*.parquet\n'
        body='''# MiniMind FineWeb-Edu Score >= 3.5 — v1.0

Exact dataset release for the completed 2026-10-24 Dense/MoE full-document pretraining experiment. This is **not the entire FineWeb-Edu corpus**: it is the existing score >= 3.5 selection from `sample/10BT` at the pinned upstream revision.

- Selected source: 1,288,228 documents. The threshold is **>= 3.5**, not > 3.5.
- Actual training: 1,096,131 documents, **4,407,066 windows**; validation: 274 documents, 1,095 windows.
- 191,823 documents shorter than one complete window across both assigned splits are in `excluded_documents`. They produced no optimizer/evaluation windows.
- BOS + entire document + EOS; 340-token windows, stride 339; final incomplete window discarded **within each document**, no cross-document packing. Training shuffled all windows for 2 epochs.
- `documents` contains full source text, including the trailing text omitted by training. The authoritative exact training inputs are `arrays/*.bin`, not all of that text. `windows`, `first_window`, `tokenized_tokens`, `dropped_tokens` and `assigned_split` make this distinction explicit.
- Split: first 8 bytes of SHA256(text), modulo 4096 equals zero → validation; otherwise train. Source order and original document index are retained.
- Tokenizer: bundled original 6,400-token tokenizer. Arrays are uint16, shape `[N,340]`. Model training shifts next-token labels internally. Evaluation originally used a fixed 1,024-window validation subset; see the experiment recipe.

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import numpy as np
repo = "caspar/MiniMind-FineWeb-Edu-Score35-v1.0"
docs = load_dataset(repo, "documents", revision="v1.0")
path = hf_hub_download(repo, "arrays/train.bin", repo_type="dataset", revision="v1.0")
windows = np.memmap(path, mode="r", dtype="<u2", shape=(4407066, 340))
```

## License and attribution

Derived from [HuggingFaceFW/FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), ODC-By 1.0, also subject to [Common Crawl terms](https://commoncrawl.org/terms-of-use). Credit the FineWeb/FineWeb-Edu creators and preserve upstream attribution and notices. Web documents remain subject to their original rights. Original conversion records, shard manifest and the upstream dataset card are included. The `attribution` config preserves original URL, document ID, dump, WARC file path, language and education score for all 1,288,228 selected documents. Join it to the documents by `source_row`. Each row was checked against the actual training JSONL text in original source order; filtered Parquet shard hashes were also verified.
'''
    elif kind=='sft':
        header+='license: other\nlicense_name: upstream-component-licenses\nlicense_link: LICENSE_NOTES.md\nconfigs:\n- config_name: conversations\n  default: true\n  data_files:\n'
        for split in count:header+=f'  - split: {split}\n    path: conversations/{split}/*.parquet\n'
        body='''# MiniMind Mixed SFT Training — v1.0

Exact retained conversations, labels and windows for the completed Dense/MoE mixed-SFT experiment of 2026-10-24/25. Both models used this same release. This does not merge older UltraChat-only or GSM8K-only experiments.

| Training source | Retained conversations |
|---|---:|
| UltraChat 200k, train_sft | 207,865 |
| SmolTalk, all/train | 1,020,912 |
| GSM8K, main/train | 6,851 |
| Total | **1,235,628** |

The exact inputs comprise **5,396,469 training windows**, 340 tokens, stride 256, overlap 84, 2 epochs. All retained conversations and final assistant/EOS targets are covered; overlap labels are counted only once. Padding ID is 0, ignored labels -100. The empty thinking block is deterministically removed from the original chat template. GSM8K questions have the original experiment's step-by-step / `####` suffix.

Validation contains 1,319 GSM8K, 512 UltraChat and 512 SmolTalk conversations from official test splits, never optimizer training data. The GSM8K validation examples also constituted the reported 1,319-question generation benchmark: they are not an additional unseen test set. Reported PPL used 512 fixed windows per source listed in `provenance/prepared.json`.

Official test prompt overlap, exact normalized conversation duplicates and normalized GSM8K test question substrings were removed from training. This does not guarantee paraphrase-level decontamination. Source row indices, component labels and original audit/window offsets are retained. `messages_json` is a JSON string so optional tool/reasoning metadata is preserved without a lossy common Arrow struct schema.

```python
import json, numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
repo = "caspar/MiniMind-Mixed-SFT-Training-v1.0"
data = load_dataset(repo, "conversations", revision="v1.0")
messages = json.loads(data["train"][0]["messages_json"])
path = hf_hub_download(repo, "arrays/train.bin", repo_type="dataset", revision="v1.0")
pairs = np.memmap(path, mode="r", dtype="<i2", shape=(5396469, 2, 340))
input_ids, labels = pairs[0]
```

## License and attribution

UltraChat 200k and GSM8K are MIT licensed. SmolTalk is a collection: newly generated Smol datasets are Apache-2.0; existing datasets retain their respective original licenses. **There is no blanket license overriding those component terms.** The complete upstream cards, composition links, component counts, pinned revisions and file hashes are included. See `LICENSE_NOTES.md` and `provenance/source-cards/` before reuse.

Models: [Dense](https://huggingface.co/caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0), [MoE](https://huggingface.co/caspar/MiniMind-MoE-198M-A64M-Mixed-SFT-v1.0).
'''
    else:
        header+='license: other\nlicense_name: upstream-llava-and-image-terms\nlicense_link: LICENSE_NOTES.md\nconfigs:\n'
        for stage in ['pretrain','sft']:
            header+=f'- config_name: {stage}\n'
            if stage=='sft':header+='  default: true\n'
            header+='  data_files:\n'
            for split in ['train','validation','test']:
                if f'{stage}_{split}' in count:header+=f'  - split: {split}\n    path: conversations/{stage}_{split}/*.parquet\n'
        body='''# MiniMind-V LLaVA Short Training — v1.0

Exact data used by [MiniMind-V Dense 64M LLaVA Short v1.0](https://huggingface.co/caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0), completed 2026-10-25. This release covers the successful short-data experiment, not the stopped full 558K/150K run or ShareGPT4V.

| Stage | Selected | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Projector alignment | 50,000 | 49,744 | 256 | 0 |
| Visual SFT (standalone QA) | 150,607 | 149,227 | 690 | 690 |

Alignment: seed 42 uniform sample of captions with **1–19 whitespace-separated words** from `blip_laion_cc_sbu_558k.json`. Visual SFT: extract standalone simple identification/conversation QA, answer 1–2 sentences by punctuation heuristic and <=40 words, question <=30 words. Exclude complex-reasoning component and matching image/QA pairs; filter speculative/context-dependent questions, uncertainty and duplicates. All retained SFT pairs came from `conversation_58k.json`; zero from detail_23k. This is a deterministic heuristic, not human annotation of simplicity.

Heldouts are grouped by SHA256 of EXIF-corrected RGB image pixels resized to 256x256 BICUBIC, including exclusion across training stages. `provenance/*-record-splits.json` specifies actual row membership. `selected/*.json` retains the full selected source order; optimizer training uses only each config's train split. See exact arrays and image-index mapping for reconstruction.

`images/{stage}/*.zip` contain **only selected original image bytes**, preserving the original filenames under each row's `image` key. These are not resized or JPEG-reencoded copies. `image_archive` identifies the ZIP shard. `provenance/*-image-archives.json` includes original archive members, byte SHA256 and resized pixel SHA256. `provenance/pretrain-image-attribution.json` preserves upstream image URLs and caption metadata; `provenance/coco-image-attribution.json` preserves selected COCO image URLs and individual license records. Text Parquet rows link to ZIP images rather than duplicating the same image for every QA pair.

```python
import io, zipfile
from PIL import Image, ImageOps
from datasets import load_dataset
from huggingface_hub import hf_hub_download
repo = "caspar/MiniMind-V-LLaVA-Short-Training-v1.0"
row = load_dataset(repo, "sft", split="test", revision="v1.0")[0]
path = hf_hub_download(repo, row["image_archive"], repo_type="dataset", revision="v1.0")
with zipfile.ZipFile(path) as z:
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(z.read(row["image"])))).convert("RGB")
print(row["conversations"])
```

Training arrays: little-endian int16 `[N,2,340]` (input IDs and assistant labels; ignore=-100); corresponding `*-images.npy` indexes `provenance/*-images.json`. Windows use the exact original recipe, tokenizer and image processor config. Images use a frozen SigLIP2 encoder; projector alignment then projector plus first/last LLM layer SFT, one epoch each. See the recipe and manifests for counts and settings.

## License and attribution — research reproduction

The LLaVA-Pretrain packaged images were provided by the upstream authors **only for research reproduction**. That condition is retained here; image rights and LAION/CC/SBU conditions remain with their owners. BLIP synthetic captions retain the applicable BLIP license. LLaVA-Instruct annotation metadata lists CC-BY-4.0; its upstream card additionally requires compliance with OpenAI terms. COCO images retain their individual original image licenses. **This release does not grant a blanket commercial image license or override upstream restrictions.** Preserve source attribution and honor upstream removal requests. The pinned upstream source cards and their original license links are included in `provenance/source-cards/`.
'''
    source_lines='\n'.join(f'- [{SOURCES[n][0]}](https://huggingface.co/datasets/{SOURCES[n][0]}/tree/{SOURCES[n][1]}) — `{SOURCES[n][1]}`' for n in source_names)
    body+='\n## Pinned sources\n\n'+source_lines+'\n\n## Integrity and reproducibility\n\n`SHA256SUMS.json` lists every released artifact except itself. Original training manifests retain historical `/shared-volume/` paths for provenance; use this repository’s relative paths when loading. `recipe/` contains the original preparation logic as reference, not a portable one-command cloud launcher. No API keys, optimizer states or model weights are included. Version: **v1.0**.\n\nExported row counts:\n\n```json\n'+json.dumps(count,indent=2)+'\n```\n'
    header=header.replace('license_link: LICENSE_NOTES.md',f'license_link: https://huggingface.co/datasets/{REPOS[kind]}/blob/main/LICENSE_NOTES.md')
    (dest/'README.md').write_text(header+'---\n'+body)
    (dest/'LICENSE_NOTES.md').write_text('# Upstream licensing and attribution\n\n'+body[body.index('## License'):body.index('## Pinned sources')]+'\n'+source_lines+'\n\nOriginal cards are reproduced in provenance/source-cards/. Their terms are not replaced by the metadata label of this derived dataset.\n')
