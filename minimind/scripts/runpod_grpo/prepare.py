import hashlib,json,os,shutil
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from huggingface_hub import HfApi,hf_hub_download,snapshot_download
root=Path(os.environ['GRPO_ROOT']);runtime=Path(os.environ['GRPO_RUNTIME'])
base=Path('/shared-volume/minimind-dpo-ultrafeedback-20261020')
for n in ['out','checkpoints','data','tokenizer','reward_model']:(root/n).mkdir(exist_ok=True)
expected=json.loads((base/'evaluation_after.json').read_text())['checkpoint_sha256']
with (base/'out/dpo_768.pth').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected
for p in (base/'tokenizer').iterdir():
    if p.is_file():shutil.copyfile(p,root/'tokenizer'/p.name)
shutil.copyfile(base/'config.json',root/'config.json')
repo=runtime/'repo'
for n in ['tokenizer.json','tokenizer_config.json']:shutil.copyfile(root/'tokenizer'/n,repo/'model'/n)
if not (repo/'out/dpo_768.pth').exists():(repo/'out/dpo_768.pth').symlink_to(base/'out/dpo_768.pth')
if not (repo/'checkpoints').exists():(repo/'checkpoints').symlink_to(root/'checkpoints',target_is_directory=True)
tok=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True)
api=HfApi();dataset='HuggingFaceH4/ultrafeedback_binarized'
revision=json.loads((base/'dataset_manifest.json').read_text())['revision']
manifest=dict(dataset=dataset,revision=revision,base_checkpoint_sha256=expected,max_prompt_tokens=768,max_completion_tokens=256,system_prompt='You are a helpful assistant.',splits={},train_sampling='All valid train_prefs prompts within context limit; no random subset',reward_model='Skywork/Skywork-Reward-V2-Qwen3-0.6B')
train_ids=set();validations=[]
for split in ['train_prefs','test_prefs']:
    counts=dict(source_rows=0,retained=0,excluded_long_prompt=0,excluded_invalid=0)
    dest=runtime/(split+'.jsonl')
    name='data/'+split+'-00000-of-00001.parquet'
    file=hf_hub_download(dataset,name,repo_type='dataset',revision=revision)
    with dest.open('w') as out:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=256):
            records=[];texts=[]
            for row in batch.to_pylist():
                counts['source_rows']+=1
                messages=row['chosen'][:-1]
                if not messages or messages[-1]['role']!='user' or messages!=row['rejected'][:-1]:counts['excluded_invalid']+=1;continue
                if messages[0]['role']!='system':messages=[dict(role='system',content='You are a helpful assistant.')]+messages
                text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,open_thinking=False)
                records.append(dict(prompt_id=row['prompt_id'],conversations=messages+[dict(role='assistant',content='')],prompt_tokens=None))
                texts.append(text)
            for rec,ids in zip(records,tok(texts,add_special_tokens=False).input_ids):
                if len(ids)>768:counts['excluded_long_prompt']+=1;continue
                rec['prompt_tokens']=len(ids);pid=rec['prompt_id']
                if split=='train_prefs':
                    if pid in train_ids:counts['duplicate_prompt_id_rows']=counts.get('duplicate_prompt_id_rows',0)+1
                    train_ids.add(pid)
                else:
                    assert pid not in train_ids
                    validations.append((hashlib.sha256(('grpo-eval42:'+pid).encode()).hexdigest(),rec))
                out.write(json.dumps(rec,ensure_ascii=False)+'\n');counts['retained']+=1
    with dest.open('rb') as f:counts['sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    manifest['splits'][split]=counts;shutil.copyfile(dest,root/'data'/dest.name)
    print(split,counts,flush=True)
validation=sorted(validations,key=lambda p:p[0])[:128];assert len(validation)==128
(runtime/'validation.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for _,r in validation));shutil.copyfile(runtime/'validation.jsonl',root/'data/validation.jsonl')
manifest['validation_prompts']=128
reward_revision=api.model_info(manifest['reward_model']).sha;manifest['reward_revision']=reward_revision
(root/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Downloading reward model',manifest['reward_model'],reward_revision,flush=True)
reward_path=snapshot_download(manifest['reward_model'],revision=reward_revision,allow_patterns=['*.json','*.safetensors','*.jinja','merges.txt','vocab.json','README.md'],local_dir=runtime/'reward_model')
for p in Path(reward_path).iterdir():
    if p.is_file():shutil.copyfile(p,root/'reward_model'/p.name)
print('PREPARATION_COMPLETE',json.dumps(manifest),flush=True)
