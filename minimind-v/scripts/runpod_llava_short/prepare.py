"""Full original JSON + packaged images, group-held-out splits, no silent row drops."""
import collections, concurrent.futures, io, json, shutil, zipfile
import numpy as np
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer
from common import *

def download():
    old=Path('/shared-volume/minimind-v-llava-20261025')
    ROOT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
    for name in ['raw','data','tokenizer','vision_encoder','out','checkpoints','samples']:(ROOT/name).mkdir(exist_ok=True)
    write(ROOT/'settings.json',SETTINGS)
    selection=json.loads((ROOT/'filter_report.json').read_text())
    for stage in ['pretrain','sft']:
        assert sha(ROOT/'raw'/f'{stage}.json')==selection['hashes'][f'{stage}.json']
    assert sha(BASE/'dense/out/full_sft_768.pth')==BASE_SHA
    shutil.copyfile(BASE/'dense/out/full_sft_768.pth',LOCAL/'base.pth')
    for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja','vocab.json','merges.txt']:
        if (BASE/'tokenizer'/name).exists():shutil.copyfile(BASE/'tokenizer'/name,ROOT/'tokenizer'/name)
    shutil.copyfile(BASE/'dense/config.json',ROOT/'base_config.json')
    hashes=json.loads((old/'raw/hashes.json').read_text())
    for stage in ['pretrain','sft']:
        status('copy_cached_images',stage=stage,bytes=(old/'raw'/f'{stage}.zip').stat().st_size)
        shutil.copyfile(old/'raw'/f'{stage}.zip',LOCAL/f'{stage}.zip')
        assert sha(LOCAL/f'{stage}.zip')==hashes[stage]
    for name in ['config.json','preprocessor_config.json','model.safetensors']:
        shutil.copyfile(old/'vision_encoder'/name,ROOT/'vision_encoder'/name)
        assert sha(old/'vision_encoder'/name)==sha(ROOT/'vision_encoder'/name)
    write(ROOT/'raw/source_images.json',dict(source_root=str(old/'raw'),hashes=hashes))

def prepare():
    tok=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True)
    assert len(tok)==6400 and tok.convert_tokens_to_ids('<|image_pad|>')==12
    all_rows={};image_records={};hashes={};mapping={}
    for stage in ['pretrain','sft']:
        rows=json.loads((ROOT/'raw'/f'{stage}.json').read_text());all_rows[stage]=rows
        # Actual row count is audited rather than trusting rounded dataset names.
        status('verify_images',stage=stage,rows=len(rows))
        with zipfile.ZipFile(LOCAL/f'{stage}.zip') as z:
            names=z.namelist();by_suffix={}
            for n in names:
                if n.endswith('/'):continue
                key='/'.join(n.split('/')[-2:]) if stage=='pretrain' else Path(n).name
                assert key not in by_suffix, key
                by_suffix[key]=n
            needed=sorted({r['image'] for r in rows});records=[]
            for i,key in enumerate(needed):
                lookup='/'.join(key.split('/')[-2:]) if stage=='pretrain' else Path(key).name
                assert lookup in by_suffix, f'Missing image {key}'
                name=by_suffix[lookup];raw=z.read(name)
                with Image.open(io.BytesIO(raw)) as im:
                    im=ImageOps.exif_transpose(im).convert('RGB').resize((256,256),Image.Resampling.BICUBIC)
                    digest=hashlib.sha256(im.tobytes()).hexdigest()
                records.append(dict(key=key,member=name,pixel_sha256=digest));mapping[(stage,key)]=i
                if i%5000==0:status('verify_images',stage=stage,verified=i,total=len(needed))
            image_records[stage]=records;hashes[stage]={r['pixel_sha256'] for r in records}
        write(ROOT/'data'/f'{stage}-images.json',records)
    # SFT validation/test held out by resized decoded pixels, across both phases.
    ordered=sorted(hashes['sft'],key=lambda h:hashlib.sha256(('42'+h).encode()).hexdigest())
    val=set(ordered[:256]);test=set(ordered[256:512])
    pt_val=set(sorted(hashes['pretrain']-val-test)[:256]);blocked=val|test|pt_val
    write(ROOT/'data/heldout.json',dict(sft_validation=sorted(val),sft_test=sorted(test),pretrain_validation=sorted(pt_val)))
    manifest={}
    for stage,rows in all_rows.items():
        streams={s:(LOCAL/f'{stage}-{s}.bin').open('wb') for s in ['train','validation','test']}
        indexes={s:[] for s in streams};counts=collections.Counter();record_splits={s:[] for s in streams}
        for i,row in enumerate(rows):
            idx=mapping[(stage,row['image'])];digest=image_records[stage][idx]['pixel_sha256']
            if stage=='sft':split='validation' if digest in val else 'test' if digest in test else 'excluded' if digest in pt_val else 'train'
            else:split='validation' if digest in pt_val else 'excluded' if digest in val|test else 'train'
            counts[split+'_rows']+=1
            if split=='excluded':continue
            record_splits[split].append(i)
            before=len(indexes[split])
            for x,y in windows(tok,row):
                streams[split].write(np.array([x,y],dtype=np.int16).tobytes());indexes[split].append(idx)
                counts[split+'_targets']+=sum(v!=-100 for v in y[1:])
            assert len(indexes[split])>before, f'No target tokens: {stage} row {i}'
            if i%2000==0:status('tokenize',stage=stage,rows=i,total=len(rows),windows=sum(map(len,indexes.values())))
        for split,f in streams.items():
            f.close();np.save(LOCAL/f'{stage}-{split}-images.npy',np.array(indexes[split],dtype=np.int32))
            counts[split+'_windows']=len(indexes[split])
            for filename in [f'{stage}-{split}.bin',f'{stage}-{split}-images.npy']:
                shutil.copyfile(LOCAL/filename,ROOT/'data'/filename)
        manifest[stage]=dict(counts,source_rows=len(rows),unique_images=len(image_records[stage]),caption_sha256=sha(ROOT/'raw'/f'{stage}.json'))
        write(ROOT/'data'/f'{stage}-record-splits.json',record_splits)
    assert not (hashes['pretrain']-blocked)&(val|test)
    write(ROOT/'dataset_manifest.json',manifest);write(ROOT/'PREPARED.json',dict(complete=True,manifest=manifest))
    status('prepared',manifest=manifest)

if __name__=='__main__':
    download();prepare()
