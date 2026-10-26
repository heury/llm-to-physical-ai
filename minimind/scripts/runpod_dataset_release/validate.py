"""Validate public loading and original tokenizer/image inputs after publication."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer
from PIL import Image,ImageOps,features
from cards import RELEASE,REPOS

def main():
    report={}
    for kind,repo in REPOS.items():
        receipt=json.loads((RELEASE/f'{kind}-published.json').read_text());root=RELEASE/'datasets'/kind
        configs={'pretrain':[('documents','train'),('attribution','source')],'sft':[('conversations','train')],'vision':[('pretrain','validation'),('sft','test')]}[kind]
        result={'commit':receipt['commit'],'public_streaming':[]}
        for config,split in configs:
            ds=load_dataset(repo,config,split=split,revision=receipt['tag'],streaming=True,token=False)
            row=next(iter(ds));assert row
            result['public_streaming'].append({'config':config,'split':split,'fields':list(row)})
        if kind in ['pretrain','sft']:
            tok=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True);assert len(tok)==6400
            if kind=='pretrain':
                for split in ['train','validation']:
                    row=pq.read_table(next((root/'documents'/split).glob('*.parquet'))).slice(0,1).to_pylist()[0]
                    ids=[1]+tok(row['text'],add_special_tokens=False).input_ids+[2]
                    arr=np.memmap(root/'arrays'/f'{split}.bin',mode='r',dtype='<u2').reshape(-1,340)
                    assert arr[row['first_window']].tolist()==ids[:340]
                result['token_windows']='train and validation match original tokenizer'
            else:
                row=pq.read_table(next((root/'conversations/train').glob('*.parquet'))).slice(0,1).to_pylist()[0]
                msgs=json.loads(row['messages_json']);text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False,open_thinking=False).replace('<think>\n\n</think>\n\n','')
                ids=tok(text,add_special_tokens=False).input_ids
                arr=np.memmap(root/'arrays/train.bin',mode='r',dtype='<i2').reshape(-1,2,340)
                expected=(ids[:340]+[0]*340)[:340];assert arr[row['first_window'],0].tolist()==expected
                assert (arr[row['first_window'],1]==-100).any();result['token_windows']='first actual SFT window matches conversation template and tokenizer'
        else:
            samples=[]
            for stage in ['pretrain','sft']:
                mapping=json.loads((root/'provenance'/f'{stage}-image-archives.json').read_text())
                for split in ['train','validation','test']:
                    d=root/'conversations'/f'{stage}_{split}'
                    if not d.exists():continue
                    row=pq.read_table(next(d.glob('*.parquet'))).slice(0,1).to_pylist()[0];record=mapping[row['image']]
                    with zipfile.ZipFile(root/record['archive']) as z:raw=z.read(row['image'])
                    assert hashlib.sha256(raw).hexdigest()==record['sha256']
                    im=ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB').resize((256,256),Image.Resampling.BICUBIC)
                    assert hashlib.sha256(im.tobytes()).hexdigest()==record['pixel_sha256']
                    samples.append(f'{stage}/{split}')
            result['original_image_and_training_pixel_hash_checks']=samples
            result['image_decoder']={'Pillow':Image.__version__,'libjpeg_turbo':features.version_feature('libjpeg_turbo')}
            assert len(json.loads((root/'provenance/pretrain-image-attribution.json').read_text()))==50000
            assert len(json.loads((root/'provenance/coco-image-attribution.json').read_text()))==55507
        report[kind]=result;print(json.dumps({kind:result}),flush=True)
    (RELEASE/'loading-validation.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
