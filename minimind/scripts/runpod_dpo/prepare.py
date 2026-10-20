"""Pinned preference-split conversion with exact native DPO mask validation."""
import hashlib,json,math,multiprocessing,os,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pyarrow.parquet as pq
from huggingface_hub import HfApi,hf_hub_download
from transformers import AutoTokenizer
from dataset.lm_dataset import DPODataset,post_processing_chat

root=Path(os.environ['DPO_ROOT']); runtime=Path(os.environ['DPO_RUNTIME'])
base=Path('/shared-volume/minimind-sft-ultrachat-200k-20261020')
for name in ['out','checkpoints','data','tokenizer']:(root/name).mkdir(exist_ok=True)
expected=json.loads((base/'evaluation_after.json').read_text())['checkpoint_sha256']
with (base/'out/full_sft_768.pth').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected
for p in (base/'tokenizer').iterdir():
    if p.is_file():shutil.copyfile(p,root/'tokenizer'/p.name)
shutil.copyfile(base/'config.json',root/'config.json')
repo=runtime/'repo'
for n in ['tokenizer.json','tokenizer_config.json']:shutil.copyfile(base/'tokenizer'/n,repo/'model'/n)
(repo/'out/full_sft_768.pth').symlink_to(base/'out/full_sft_768.pth')
(repo/'checkpoints').symlink_to(root/'checkpoints',target_is_directory=True)
tokenizer=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True)
probe=DPODataset.__new__(DPODataset);probe.tokenizer=tokenizer;probe.max_length=1024
probe.bos_id=tokenizer(tokenizer.bos_token+'assistant\n',add_special_tokens=False).input_ids
probe.eos_id=tokenizer(tokenizer.eos_token+'\n',add_special_tokens=False).input_ids

def convert(row):
    record={'prompt_id':row['prompt_id']}
    for key in ['chosen','rejected']:
        m=row[key]
        if len(m)<2 or m[-1]['role']!='assistant' or any(x['role'] not in ['system','user','assistant'] or not isinstance(x['content'],str) for x in m):return 'invalid',None
        if m[0]['role']!='system':m=[dict(role='system',content='You are a helpful assistant.')]+m
        record[key]=m
    if record['chosen'][:-1]!=record['rejected'][:-1]:return 'prompt_mismatch',None
    for remove in [False,True]:
        ids=[]
        for key in ['chosen','rejected']:
            text=post_processing_chat(tokenizer.apply_chat_template(record[key],tokenize=False,add_generation_prompt=False),remove_empty_think=remove)
            seq=tokenizer(text,truncation=True,max_length=1024,padding='max_length').input_ids
            mask=probe.generate_loss_mask(seq)
            if sum(mask[1:])<32:return 'short_target',None
            if any(active and token==tokenizer.pad_token_id for token,active in zip(seq,mask)):return 'masked_padding',None
            ids.append(seq)
        if ids[0]==ids[1]:return 'identical_visible',None
    return 'retained',record

api=HfApi();dataset='HuggingFaceH4/ultrafeedback_binarized';revision=api.dataset_info(dataset).sha
files=api.list_repo_files(dataset,repo_type='dataset',revision=revision)
manifest=dict(dataset=dataset,revision=revision,base_checkpoint_sha256=expected,max_seq_len=1024,min_response_tokens=32,splits={},system_prompt='You are a helpful assistant.',ties='Keep official chosen/rejected labels, including score ties')
train_ids=set();validation=[]
with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:
    for split in ['train_prefs','test_prefs']:
        counts={};selected=sorted(n for n in files if n.startswith('data/'+split+'-') and n.endswith('.parquet'));assert selected
        dest=runtime/(split+'.jsonl')
        with dest.open('w') as out:
            for name in selected:
                file=hf_hub_download(dataset,name,repo_type='dataset',revision=revision)
                for batch in pq.ParquetFile(file).iter_batches(batch_size=256):
                    for kind,row in pool.map(convert,batch.to_pylist(),chunksize=16):
                        counts[kind]=counts.get(kind,0)+1
                        if row is None:continue
                        pid=row['prompt_id']
                        if split=='train_prefs':train_ids.add(pid)
                        else:
                            assert pid not in train_ids
                            validation.append((hashlib.sha256(('dpo-eval42:'+pid).encode()).hexdigest(),row))
                        out.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(split,name,counts,flush=True)
        shutil.copyfile(dest,root/'data'/dest.name)
        with dest.open('rb') as f:counts['sha256']=hashlib.file_digest(f,'sha256').hexdigest()
        counts['files']=selected;manifest['splits'][split]=counts
rows=sorted(validation,key=lambda x:x[0])[:512];assert len(rows)==512
(runtime/'validation.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for _,r in rows))
shutil.copyfile(runtime/'validation.jsonl',root/'data/validation.jsonl')
manifest['validation_pairs']=512;manifest['steps_per_epoch']=math.ceil(manifest['splits']['train_prefs']['retained']/32)
(root/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
settings=dict(epochs=1,batch_size=32,accumulation_steps=2,learning_rate=5e-7,beta=0.1,dtype='bfloat16',max_seq_len=1024,hidden_size=768,num_hidden_layers=8,log_interval=10,save_interval=250)
(root/'training_settings.json').write_text(json.dumps(settings,indent=2))
print(json.dumps(manifest),flush=True)
