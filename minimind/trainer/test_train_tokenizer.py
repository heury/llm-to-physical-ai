"""Offline checks for local and streamed tokenizer training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trainer.train_tokenizer import get_hf_texts, get_texts, train_tokenizer


class TrainingTests(unittest.TestCase):
    def test_hf_stream_and_limit(self):
        rows = [{'text': ' '}, {'text': '한국어 학습'}, {'text': 'English text'}]
        with patch('datasets.load_dataset', return_value=iter(rows)) as load:
            texts = list(get_hf_texts('example/corpus', revision='fixed', max_lines=1))
        self.assertEqual(texts, ['한국어 학습'])
        load.assert_called_once_with('example/corpus', split='train', revision='fixed', streaming=True)

    def test_empty_stream_fails(self):
        with patch('datasets.load_dataset', return_value=iter([{'text': None}])):
            with self.assertRaises(ValueError):
                list(get_hf_texts('example/empty'))

    def test_stream_accounts_for_empty_rows(self):
        stats = {}
        rows = [{'text': '한국어'}, {'text': '  '}, {'text': 'English'}]
        with patch('datasets.load_dataset', return_value=iter(rows)):
            self.assertEqual(list(get_hf_texts('example/corpus', stats=stats)), ['한국어', 'English'])
        self.assertEqual(stats, {'dataset_rows': 3, 'skipped_empty_rows': 1})

    def test_local_formats_and_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data.jsonl'
            data.write_text('\n'.join(json.dumps(row) for row in [
                {'text': '안녕하세요. Hello! ' * 50},
                {'conversations': [{'content': '한국어 토크나이저 학습입니다. ' * 50}]},
            ]), encoding='utf-8')
            self.assertEqual(len(list(get_texts(data))), 2)
            out = root / 'tokenizer'
            train_tokenizer(data, out, 400, texts=get_texts(data))
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(out, local_files_only=True)
            self.assertGreater(len(tokenizer), 36 + 256)
            self.assertLessEqual(len(tokenizer), 400)
            self.assertEqual((tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id), (0, 1, 2))
            text = '한국어와 English 😊\n\t테스트'
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            info = json.loads((out / 'training_info.json').read_text())
            self.assertEqual(info['documents'], 2)
            from scripts.runpod_tokenizer.validate import validate
            with self.assertRaisesRegex(ValueError, 'Incomplete corpus'):
                validate(out)
            report = validate(out, expected_documents=2, expected_vocab=len(tokenizer))
            self.assertEqual(report['status'], 'passed')
            info.update({'dataset_rows': 3, 'skipped_empty_rows': 1})
            (out / 'training_info.json').write_text(json.dumps(info))
            report = validate(out, expected_documents=3, expected_vocab=len(tokenizer))
            self.assertEqual(report['skipped_empty_rows'], 1)


if __name__ == '__main__':
    unittest.main()
