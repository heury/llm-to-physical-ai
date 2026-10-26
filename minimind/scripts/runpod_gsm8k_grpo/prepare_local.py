"""Extract full GSM8K conversations from hash-verified release artifacts."""
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent / 'outputs/dataset-release-v1.0/datasets/sft'
OUT = PROJECT / 'outputs/gsm8k-grpo-20261025'

def main():
    manifest = json.loads((ROOT/'SHA256SUMS.json').read_text())
    stats, questions = {}, {}
    for split, filename in [('train', 'conversations/train/00000.parquet'),
                            ('test', 'conversations/validation_gsm8k/00000.parquet')]:
        p = ROOT/filename
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest == manifest[filename]['sha256']
        rows = []
        for row in pq.read_table(p).to_pylist():
            if row['dataset'] != 'gsm8k':
                continue
            messages = json.loads(row['messages_json'])
            assert [m['role'] for m in messages] == ['user', 'assistant']
            rows.append(dict(index=row['index'], question=messages[0]['content'], answer=messages[1]['content']))
        assert len(rows) == (6851 if split == 'train' else 1319)
        target = OUT/(split+'.jsonl')
        target.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        questions[split] = {r['question'] for r in rows}
        stats[split] = dict(rows=len(rows), source=filename, source_sha256=digest,
                            sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    assert not questions['train'] & questions['test']
    result = dict(repo='caspar/MiniMind-Mixed-SFT-Training-v1.0', revision='542384d27d374d668413d8ade5b9973b7b063bee',
                  splits=stats, exact_question_overlap=0)
    (OUT/'dataset_manifest.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))

if __name__ == '__main__':
    main()
