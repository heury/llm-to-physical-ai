"""Verify complete-corpus MiniMind tokenizer training before downloading artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def validate(directory, expected_documents=1284879, expected_vocab=6400):
    directory = Path(directory)
    info = json.loads((directory / 'training_info.json').read_text(encoding='utf-8'))
    skipped = info.get('skipped_empty_rows', 0)
    total = info['documents'] + skipped
    if skipped < 0 or total != expected_documents or info.get('dataset_rows', total) != total:
        raise ValueError(f"Incomplete corpus: {info['documents']} usable + {skipped} empty != {expected_documents}")
    if info.get('max_lines', 0) != 0:
        raise ValueError('Training used a document limit')
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    if len(tokenizer) != expected_vocab:
        raise ValueError(f'Unexpected vocabulary: {len(tokenizer)} != {expected_vocab}')
    if (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id) != (0, 1, 2):
        raise ValueError('MiniMind PAD/BOS/EOS IDs changed')
    samples = [
        '안녕하세요. 한국어 웹 문서로 학습한 토크나이저입니다.',
        'Hello! This tokenizer can encode English text.',
        '한국어와 English, 숫자 123 및 기호 !@#$% 😊',
        'def hello(name):\n    return f"안녕, {name}!"\n\t공백  테스트',
        '한글 ㄱㄴㄷ 中文 日本語 العربية 🚀',
    ]
    syllables = ''.join(chr(code) for code in range(0xac00, 0xd7a4))
    samples.extend(syllables[i:i + 64] for i in range(0, len(syllables), 64))
    counts = []
    for text in samples:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if tokenizer.decode(ids) != text:
            raise ValueError(f'Round-trip mismatch: {text[:60]!r}')
        if not all(0 <= token_id < expected_vocab for token_id in ids):
            raise ValueError('Out-of-range token ID')
        counts.append(len(ids))
    messages = [{'role': 'user', 'content': '안녕하세요.'},
                {'role': 'assistant', 'content': 'Hello! 반갑습니다.'}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    ids = tokenizer.apply_chat_template(messages, tokenize=True)
    if tokenizer.decode(ids) != prompt or ids.count(tokenizer.eos_token_id) != 2:
        raise ValueError('Chat template round-trip or boundaries failed')
    if tokenizer.bos_token + 'assistant\n' not in prompt:
        raise ValueError('MiniMind SFT assistant boundary is missing')
    raw = json.loads((directory / 'tokenizer.json').read_text(encoding='utf-8'))
    if not raw['model']['merges']:
        raise ValueError('Tokenizer has no learned BPE merges')
    artifact_names = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json',
                      'merges.txt', 'training_info.json']
    if (directory / 'corpus_audit.json').exists():
        artifact_names.append('corpus_audit.json')
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
              for name in artifact_names}
    report = {'status': 'passed', 'documents': info['documents'],
              'dataset_rows': total, 'skipped_empty_rows': skipped,
              'vocab_size': len(tokenizer), 'round_trip_samples': len(samples),
              'modern_hangul_syllables_checked': len(syllables),
              'example_token_counts': counts[:5], 'sha256': hashes}
    (directory / 'validation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--expected-documents', type=int, default=1284879)
    parser.add_argument('--expected-vocab', type=int, default=6400)
    args = parser.parse_args()
    print(json.dumps(validate(args.directory, args.expected_documents, args.expected_vocab), indent=2))
