import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dataset.gsm8k_dataset import GSM8KDataset
from trainer.gsm8k_reward import score_gsm8k_response


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['open_thinking'] is False
        return messages[0]['content'] + '\nassistant:'

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=list(range(len(text))))


class GSM8KIntegrationTests(unittest.TestCase):
    def load(self, rows, max_length=1000):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            return GSM8KDataset(str(path), Tokenizer(), max_length=max_length)

    def test_original_gold_not_in_prompt(self):
        ds = self.load([dict(question='How many?', answer='SECRET rationale\n#### 72')])
        self.assertNotIn('SECRET', ds[0]['prompt'])
        self.assertNotIn('72', ds[0]['prompt'])
        self.assertEqual(ds[0]['answer'], 'SECRET rationale\n#### 72')

    def test_mixed_dataset_component_filter(self):
        messages = json.dumps([dict(role='user', content='How many?'),
                               dict(role='assistant', content='#### 72')])
        ds = self.load([dict(dataset='ultrachat', split='train', messages_json=messages),
                        dict(dataset='gsm8k', split='train', messages_json=messages)])
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]['answer'], '#### 72')

    def test_no_question_truncation(self):
        with self.assertRaisesRegex(ValueError, 'exceeds max_seq_len'):
            self.load([dict(question='A long question', answer='#### 72')], max_length=1)

    def test_no_validation_training(self):
        with self.assertRaisesRegex(ValueError, 'Only training'):
            self.load([dict(question='Q', answer='#### 72', split='test')])

    def test_response_diagnostics(self):
        for text in ['text####72', '<think>#### 72', '<think>#### 72</think>\n#### 72',
                     '#### 72.', '#### 72\n#### 72']:
            self.assertEqual(score_gsm8k_response(text, '#### 72'), (0., 'format_error'))
        self.assertEqual(score_gsm8k_response('<think>work</think>\n#### 72', '#### 72'), (1., 'correct'))
        self.assertEqual(score_gsm8k_response('#### 73', '#### 72'), (0., 'wrong_number'))
