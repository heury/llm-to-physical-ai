"""Sample-based checks using the real local tokenizer and JSONL loader."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.lm_dataset import PretrainDataset


class PretrainDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            Path(__file__).resolve().parents[1] / 'model', local_files_only=True,
        )

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.data_path = Path(temp.name) / 'samples.jsonl'
        self.texts = ['Hello!', '안녕하세요. MiniMind 테스트 😊', '', '긴 문장입니다. ' * 100]
        self.data_path.write_text(
            '\n'.join(json.dumps({'text': text}, ensure_ascii=False) for text in self.texts),
            encoding='utf-8',
        )
        self.dataset = PretrainDataset(str(self.data_path), self.tokenizer, max_length=32)

    def test_short_text_boundaries_padding_and_labels(self):
        input_ids, labels = self.dataset[0]
        content = self.tokenizer.encode(self.texts[0], add_special_tokens=False)
        expected = [self.tokenizer.bos_token_id, *content, self.tokenizer.eos_token_id]
        self.assertEqual(len(self.dataset), len(self.texts))
        self.assertEqual(input_ids.dtype, torch.long)
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(input_ids.shape, (32,))
        self.assertEqual(labels.shape, (32,))
        self.assertEqual(input_ids[:len(expected)].tolist(), expected)
        self.assertTrue(torch.all(input_ids[len(expected):] == self.tokenizer.pad_token_id))
        self.assertEqual(labels[:len(expected)].tolist(), expected)
        self.assertTrue(torch.all(labels[len(expected):] == -100))
        # Labels retain their positions; the causal LM applies the next-token shift.
        labels[0] = -100
        self.assertEqual(input_ids[0].item(), self.tokenizer.bos_token_id)

    def test_multilingual_text_round_trip(self):
        dataset = PretrainDataset(str(self.data_path), self.tokenizer, max_length=128)
        input_ids, labels = dataset[1]
        length = (labels != -100).sum().item()
        self.assertEqual(self.tokenizer.decode(input_ids[1:length - 1]), self.texts[1])

    def test_empty_text_keeps_bos_and_eos(self):
        input_ids, labels = self.dataset[2]
        expected = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]
        self.assertEqual(input_ids[:2].tolist(), expected)
        self.assertEqual(labels[:2].tolist(), expected)
        self.assertTrue(torch.all(labels[2:] == -100))

    def test_truncation_preserves_eos(self):
        input_ids, labels = self.dataset[3]
        content = self.tokenizer.encode(self.texts[3], add_special_tokens=False)
        self.assertGreater(len(content), 30)
        self.assertEqual(input_ids.tolist(), [self.tokenizer.bos_token_id, *content[:30], self.tokenizer.eos_token_id])
        self.assertTrue(torch.equal(input_ids, labels))

    def test_dataloader_batch(self):
        input_ids, labels = next(iter(DataLoader(self.dataset, batch_size=4)))
        self.assertEqual(input_ids.shape, (4, 32))
        self.assertEqual(labels.shape, (4, 32))
        padding = input_ids == self.tokenizer.pad_token_id
        self.assertTrue(torch.all(labels[padding] == -100))
        self.assertTrue(torch.equal(labels[~padding], input_ids[~padding]))


if __name__ == '__main__':
    unittest.main()
