"""Pinned Typed Decisions -> native parallel MiniMind-J; GPU training and evaluation."""
import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import urllib.request

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset import DecisionCollator, answer_values, validate_row
from model.decision import DecisionModel, load_checkpoint, save_checkpoint
from model.model_minimind import MiniMindConfig

DATASET = 'LocalLLaMA/typed-decisions'
REVISION = 'd9eda7f31ba227333213a1787f2c78fdaa4090c8'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def convert_record(raw):
    questions = json.loads(raw['questions']) if isinstance(raw['questions'], str) else raw['questions']
    gold = json.loads(raw['gold']) if isinstance(raw['gold'], str) else raw['gold']
    state = json.loads(raw['state']) if isinstance(raw['state'], str) else raw['state']
    result = {'id': raw['id'], 'workflow': raw['workflow'],
              'input': json.dumps(state, ensure_ascii=False, sort_keys=True), 'questions': []}
    for qid, question in questions.items():
        kind, criteria = question['type'], question.get('criteria', {})
        q = {'id': qid, 'type': kind, 'question': question['instructions']}
        if kind == 'choice':
            q['choices'] = list(criteria)
            rubric = '\n'.join(f'{key}: {value}' for key, value in criteria.items())
            label = str(gold[qid]['label'])
        elif kind == 'score':
            q.update(min=0, max=len(criteria)-1)
            rubric = '\n'.join(f'{level}: {description}' for level, description in enumerate(criteria))
            label = int(gold[qid]['label'])
        elif kind == 'noul':
            rubric = '\n'.join(f'{key}: {value}' for key, value in criteria.items())
            if str(gold[qid]['label']).lower() not in ('true', 'false'):
                raise ValueError('Invalid noul label')
            label = str(gold[qid]['label']).lower() == 'true'
        else:
            raise ValueError(f'Unsupported source type: {kind}')
        q['question'] += '\nRubric:\n' + rubric
        q['answer'] = label
        keys = [('true' if value else 'false') if type(value) is bool else str(value) for value in answer_values(q)]
        probabilities = gold[qid]['probabilities']
        if set(keys) != set(probabilities):
            raise ValueError(f'{qid}: gold probability keys do not match schema')
        values = [float(probabilities[key]) for key in keys]
        if any(not math.isfinite(v) or v < 0 for v in values) or not abs(sum(values)-1) < 1e-4:
            raise ValueError('Invalid source probability distribution')
        q['target_probabilities'] = [v / sum(values) for v in values]
        result['questions'].append(q)
    validate_row(result)
    return result


def split_training(rows, fraction, seed):
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['workflow']].append(row)
    train, validation = [], []
    for name in sorted(groups):
        group = sorted(groups[name], key=lambda row: row['id'])
        rng.shuffle(group)
        count = max(1, round(len(group) * fraction))
        if count >= len(group):
            raise ValueError('Not enough rows for train/validation split')
        validation.extend(group[:count])
        train.extend(group[count:])
    return train, validation


def soft_cross_entropy(logits, targets, mask):
    # Replacing masked log probabilities prevents the undefined 0 * -inf product.
    log_p = logits.float().log_softmax(-1).masked_fill(~mask, 0)
    return -(targets * log_p).sum(-1).mean()


def normalize_question_gradients(parameters, question_count):
    """Normalize summed losses once per accumulation window, before clipping."""
    if question_count < 1:
        raise ValueError('An accumulation window must contain questions')
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(question_count)


def metric_summary(items):
    def summary(entries):
        hits, tie_hits, kls, briers, confs = [], [], [], [], []
        for item in entries:
            p, g = item['p'], item['g']
            pred = max(range(len(p)), key=lambda i: p[i])
            gold = max(range(len(g)), key=lambda i: g[i])
            hits.append(int(pred == gold))
            tie_hits.append(int(abs(g[pred] - max(g)) < 1e-7))
            kls.append(sum(v * math.log(v / max(p[i], 1e-9)) for i, v in enumerate(g) if v > 0))
            briers.append(sum((a-b)**2 for a,b in zip(p,g)))
            confs.append(max(p))
        n = len(entries)
        ece = 0.0
        for b in range(10):
            indices = [i for i,c in enumerate(confs) if min(int(c*10),9)==b]
            if indices:
                ece += len(indices)/n * abs(sum(hits[i] for i in indices)/len(indices)-sum(confs[i] for i in indices)/len(indices))
        return {'decisions': n, 'accuracy': sum(hits)/n, 'tie_aware_accuracy': sum(tie_hits)/n,
                'kl_from_gold': sum(kls)/n, 'brier': sum(briers)/n, 'ece': ece}
    result = summary(items)
    for field in ('type', 'workflow'):
        result['by_'+field] = {key: summary([item for item in items if item[field]==key]) for key in sorted({item[field] for item in items})}
    return result


class SoftCollator:
    def __init__(self, tokenizer, max_length):
        self.collator = DecisionCollator(tokenizer, max_length)

    def __call__(self, rows):
        batch = self.collator(rows)
        targets = torch.zeros_like(batch['option_mask'], dtype=torch.float32)
        cursor = 0
        for row in rows:
            for q in row['questions']:
                probs = q['target_probabilities']
                targets[cursor, :len(probs)] = torch.tensor(probs)
                cursor += 1
        return {'model': batch, 'targets': targets, 'rows': rows}


def device_batch(batch):
    return {key: value.to('cuda', non_blocking=True) for key,value in batch.items()}


def evaluate(model, loader, predictions_path=None):
    model.eval()
    items, records, total_loss, count = [], [], 0.0, 0
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            tensors = device_batch(batch['model'])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(**tensors)
            probs = output['probabilities'].cpu().tolist()
            loss = soft_cross_entropy(output['logits'], batch['targets'].to('cuda'), tensors['option_mask'])
            total_loss += loss.item() * len(probs)
            count += len(probs)
            cursor = 0
            for row in batch['rows']:
                predicted = {'id': row['id'], 'workflow': row['workflow'], 'answers': {}}
                for q in row['questions']:
                    values = answer_values(q)
                    p = probs[cursor][:len(values)]
                    entry = {'type': q['type'], 'workflow': row['workflow'], 'question': q['id'],
                             'p': p, 'g': q['target_probabilities']}
                    items.append(entry)
                    predicted['answers'][q['id']] = {'value': values[max(range(len(p)), key=lambda i:p[i])],
                                                     'type': q['type'], 'values': values, 'probabilities': p,
                                                     'gold_probabilities': q['target_probabilities']}
                    cursor += 1
                records.append(predicted)
    torch.cuda.synchronize()
    result = metric_summary(items)
    result.update(loss=total_loss/count, records=len(records), seconds=time.monotonic()-started)
    if predictions_path:
        Path(predictions_path).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records))
    return result


def prior_metrics(train_rows, test_rows):
    sums = {}
    for row in train_rows:
        for q in row['questions']:
            key = (row['workflow'],q['id'])
            values = sums.setdefault(key, [0.0]*len(answer_values(q)))
            for i,p in enumerate(q['target_probabilities']): values[i] += p
    items = []
    for row in test_rows:
        for q in row['questions']:
            prior = sums[(row['workflow'],q['id'])]
            prior = [p/sum(prior) for p in prior]
            items.append({'type':q['type'],'workflow':row['workflow'],'question':q['id'],'p':prior,'g':q['target_probabilities']})
    return metric_summary(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--base-root', default='/shared-volume/minimind-sft-ultrachat-200k-20261020')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=4096)
    args = parser.parse_args()
    root, base = Path(args.root), Path(args.base_root)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'SUCCESS').exists(): raise RuntimeError('This output already completed')
    torch.manual_seed(42); random.seed(42); torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    props = torch.cuda.get_device_properties(0)
    assert 30 * 1024**3 <= props.total_memory < 40 * 1024**3, props
    started = time.monotonic()
    write_json(root/'hardware.json', {'gpu':props.name,'memory_bytes':props.total_memory,
               'torch':torch.__version__,'cuda':torch.version.cuda,'device_capability':torch.cuda.get_device_capability(0)})
    # Public pinned parquet downloads: only `all` avoids counting workflow subsets twice.
    import pyarrow.parquet as pq
    raw_dir = root/'data/raw'; raw_dir.mkdir(parents=True,exist_ok=True)
    raw = {}
    for split in ('train','test'):
        path = raw_dir/(split+'.parquet')
        url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/all/{split}-00000-of-00001.parquet'
        if not path.exists(): urllib.request.urlretrieve(url,path)
        raw[split] = pq.read_table(path).to_pylist()
    official_train = [convert_record(row) for row in raw['train']]
    test = [convert_record(row) for row in raw['test']]
    train, validation = split_training(official_train,.1,42)
    sets = [set(row['id'] for row in split) for split in (train,validation,test)]
    states = [set(row['input'] for row in split) for split in (train,validation,test)]
    assert all(not sets[i]&sets[j] and not states[i]&states[j] for i,j in [(0,1),(0,2),(1,2)])
    for name,rows in [('train',train),('validation',validation),('test',test)]:
        (root/'data'/f'{name}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    config = MiniMindConfig(**json.loads((base/'config.json').read_text()))
    tokenizer = AutoTokenizer.from_pretrained(base/'tokenizer',local_files_only=True)
    assert len(tokenizer)==config.vocab_size
    base_path = base/'out/full_sft_768.pth'
    base_hash = sha256(base_path)
    model = DecisionModel(config)
    model.load_lm_state_dict(torch.load(base_path,map_location='cpu',weights_only=True))
    model.to('cuda')
    collator = SoftCollator(tokenizer,args.max_length)
    lengths = [int(collator([row])['model']['input_ids'].shape[1]) for row in official_train+test]
    manifest = {'dataset':DATASET,'revision':REVISION,'subset':'all','split_seed':42,
                'records':{'official_train':len(official_train),'train':len(train),'validation':len(validation),'test':len(test)},
                'questions':{name:sum(len(row['questions']) for row in rows) for name,rows in [('train',train),('validation',validation),('test',test)]},
                'source_sha256':{name:sha256(raw_dir/(name+'.parquet')) for name in raw},
                'max_packed_tokens':max(lengths),'no_id_or_state_overlap':True,'base':str(base_path),'base_sha256':base_hash}
    write_json(root/'dataset_manifest.json',manifest)
    write_json(root/'settings.json',vars(args))
    print(json.dumps({'phase':'prepared','manifest':manifest}),flush=True)
    # Real longest-record backward and optimizer smoke uses disposable weights.
    longest = sorted(train,key=lambda row:len(collator([row])['model']['input_ids'][0]),reverse=True)[:args.batch_size]
    smoke = collator(longest);tensors = device_batch(smoke['model'])
    torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    with torch.autocast('cuda',dtype=torch.bfloat16): output=model(**tensors)
    loss = soft_cross_entropy(output['logits'],smoke['targets'].to('cuda'),tensors['option_mask'])
    assert torch.isfinite(loss)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    write_json(root/'smoke.json',{'loss':loss.item(),'grad_norm':float(grad),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'longest_batch_tokens':tensors['input_ids'].shape[1]})
    # Reload original source after smoke so no disposable optimization enters production.
    model.load_lm_state_dict(torch.load(base_path,map_location='cpu',weights_only=True))
    del optimizer, smoke, tensors, output, loss
    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(train,batch_size=args.batch_size,shuffle=True,generator=generator,collate_fn=collator,num_workers=0)
    val_loader = DataLoader(validation,batch_size=args.batch_size,collate_fn=collator)
    baseline = evaluate(model,val_loader)
    write_json(root/'validation_before.json',baseline)
    best_kl, best_epoch, stale, step = float('inf'),0,0,0
    history = []
    log = (root/'steps.jsonl').open('a',buffering=1)
    total_updates = args.epochs * math.ceil(len(loader)/args.accumulation)
    for epoch in range(1,args.epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        train_loss, decisions, window_questions = 0.0,0,0
        epoch_start = time.monotonic()
        for micro,batch in enumerate(loader):
            tensors = device_batch(batch['model'])
            with torch.autocast('cuda',dtype=torch.bfloat16): output=model(**tensors)
            loss = soft_cross_entropy(output['logits'],batch['targets'].to('cuda'),tensors['option_mask'])
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            questions = len(batch['targets'])
            (loss * questions).backward()
            window_questions += questions
            train_loss += loss.item()*len(batch['targets']); decisions += len(batch['targets'])
            if (micro+1)%args.accumulation==0 or micro+1==len(loader):
                normalize_question_gradients(model.parameters(), window_questions)
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
                window_questions = 0
                lr=args.lr*(.1+.9*.5*(1+math.cos(math.pi*step/total_updates)))
                for group in optimizer.param_groups:group['lr']=lr
                record={'epoch':epoch,'micro':micro+1,'step':step,'loss':loss.item(),'lr':lr,'grad_norm':float(grad),'elapsed_seconds':time.monotonic()-started}
                log.write(json.dumps(record)+'\n')
                if step%10==0:
                    write_json(root/'progress.json',record);print(json.dumps(record),flush=True)
        metrics=evaluate(model,val_loader)
        record={'epoch':epoch,'train_loss':train_loss/decisions,'validation':metrics,'seconds':time.monotonic()-epoch_start}
        history.append(record);write_json(root/'history.json',history)
        print(json.dumps(record),flush=True)
        if metrics['kl_from_gold'] < best_kl-1e-5:
            best_kl,best_epoch,stale=metrics['kl_from_gold'],epoch,0
            save_checkpoint(root/'model',model,tokenizer,args.max_length)
        else:stale+=1
        write_json(root/'progress.json',{'epoch':epoch,'step':step,'best_epoch':best_epoch,'best_validation_kl':best_kl,'stale_epochs':stale})
        if stale>=args.patience:break
    log.close()
    del optimizer,model
    torch.cuda.empty_cache()
    model,restored_tokenizer,max_length=load_checkpoint(root/'model','cuda')
    test_loader=DataLoader(test,batch_size=args.batch_size,collate_fn=SoftCollator(restored_tokenizer,max_length))
    test_metrics=evaluate(model,test_loader,root/'test_predictions.jsonl')
    train_metrics=evaluate(model,DataLoader(train,batch_size=args.batch_size,collate_fn=collator))
    prior=prior_metrics(train,test)
    check=collator(test[:1]);check_tensors=device_batch(check['model'])
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): a=model(**check_tensors)['logits']
    second,_,_=load_checkpoint(root/'model','cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): b=second(**check_tensors)['logits']
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert base_hash==sha256(base_path),'Source LM changed'
    report={'dataset':DATASET,'revision':REVISION,'mode':'specialist, full finetuning from existing MiniMind SFT',
            'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,'epochs_run':len(history),'optimizer_steps':step,
            'validation_before':baseline,'best_validation':history[best_epoch-1]['validation'],
            'train':train_metrics,'test':test_metrics,'train_only_prior_test':prior,
            'reload_logits_exact':True,'base_unchanged':True,'total_seconds':time.monotonic()-started,
            'gpu_peak_bytes':torch.cuda.max_memory_allocated()}
    write_json(root/'report.json',report)
    write_json(root/'requirements-resolved.json',json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json'])))
    paths=[p for p in root.rglob('*') if p.is_file() and p.name not in ['SHA256.json','SUCCESS','training.log','exit_code'] and not p.name.endswith('.tmp')]
    hashes={str(p.relative_to(root)):sha256(p) for p in paths}
    write_json(root/'SHA256.json',hashes)
    assert all(sha256(root/name)==value for name,value in hashes.items())
    (root/'SUCCESS').write_text(json.dumps({'verified_files':len(hashes),'model_sha256':hashes['model/decision.pth'],'best_epoch':best_epoch})+'\n')
    print(json.dumps({'phase':'SUCCESS','report':report}),flush=True)


if __name__=='__main__':
    main()
