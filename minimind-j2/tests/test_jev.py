import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from transformers import AutoTokenizer

from jev import (ROOT, Jev, OPTIONS, fit_temperature, freeze_backbone, load_jev,
                 metrics, option_ids, prompt, read_data, save_checkpoint, validate)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def tiny_model(moe=False):
    return MiniMindForCausalLM(MiniMindConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=6400,
        max_position_embeddings=512, dropout=0.2, use_moe=moe,
        num_experts=2, num_experts_per_tok=1,
    ))


class JevTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
        cls.rows = json.loads((ROOT / 'data/input.json').read_text())

    def test_padding_and_constrained_logits(self):
        for moe in (False, True):
            runner = Jev(tiny_model(moe), self.tokenizer, 512)
            batch = runner.logits(self.rows)
            for i, row in enumerate(self.rows):
                torch.testing.assert_close(batch[i], runner.logits([row])[0], atol=2e-6, rtol=2e-5)
                ids = self.tokenizer(prompt(self.tokenizer, row), add_special_tokens=False, return_tensors='pt')['input_ids']
                expected = runner.model(ids).logits[0, -1, runner.ids]
                torch.testing.assert_close(batch[i], expected, atol=2e-6, rtol=2e-5)

    def test_only_head_updates_and_loss_decreases(self):
        torch.manual_seed(7)
        runner = Jev(tiny_model(), self.tokenizer, 512)
        self.assertEqual(runner.model.lm_head.weight.data_ptr(), runner.model.model.embed_tokens.weight.data_ptr())
        freeze_backbone(runner.model)
        self.assertNotEqual(runner.model.lm_head.weight.data_ptr(), runner.model.model.embed_tokens.weight.data_ptr())
        before = {k: v.clone() for k, v in runner.model.state_dict().items()}
        optimizer = torch.optim.Adam(runner.model.lm_head.parameters(), lr=0.03)
        labels = torch.tensor([0, 1])
        initial = float(torch.nn.functional.cross_entropy(runner.logits(self.rows), labels).detach())
        for _ in range(12):
            loss = torch.nn.functional.cross_entropy(runner.logits(self.rows), labels)
            optimizer.zero_grad()
            loss.backward()
            self.assertTrue(all(p.grad is None for p in runner.model.model.parameters()))
            optimizer.step()
        final = float(torch.nn.functional.cross_entropy(runner.logits(self.rows), labels).detach())
        self.assertLess(final, initial)
        for name, tensor in runner.model.state_dict().items():
            if name != 'lm_head.weight':
                self.assertTrue(torch.equal(tensor, before[name]), name)
        untouched = [i for i in range(6400) if i not in runner.ids]
        self.assertTrue(torch.equal(runner.model.lm_head.weight[untouched], before['lm_head.weight'][untouched]))
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(runner, tmp, {})
            loaded = load_jev(argparse.Namespace(checkpoint=tmp, max_length=None, device='cpu'))
            self.assertFalse(loaded.model.config.tie_word_embeddings)
            torch.testing.assert_close(runner.logits(self.rows), loaded.logits(self.rows))

    def test_validation_and_no_label_leakage(self):
        row = dict(self.rows[0], answer='A')
        other = dict(row, answer='E')
        self.assertEqual(prompt(self.tokenizer, row), prompt(self.tokenizer, other))
        for broken in ({}, dict(row, answer='F'), dict(row, question='')):
            with self.assertRaises(ValueError):
                validate(broken, True)
        with self.assertRaises(ValueError):
            Jev(tiny_model(), self.tokenizer, 1).logits(self.rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'empty.json'
            path.write_text('[]')
            with self.assertRaises(ValueError):
                read_data(path)
        class BrokenTokenizer:
            def encode(self, *args, **kwargs):
                return [1, 2]
        with self.assertRaises(ValueError):
            option_ids(BrokenTokenizer())

    def test_calibration(self):
        logits = torch.tensor([[9., 1, 0, 0, 0], [8., 1, 0, 0, 0], [0., 8, 1, 0, 0]])
        labels = torch.tensor([0, 1, 2])
        temperature = fit_temperature(logits, labels)
        self.assertGreater(temperature, 0)
        before, after = metrics(logits, labels), metrics(logits, labels, temperature)
        self.assertLessEqual(after['nll'], before['nll'])
        self.assertEqual(before['accuracy'], after['accuracy'])
        for invalid in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                metrics(logits, labels, invalid)

    def test_cli_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model = tiny_model()
            model.config.to_json_file(tmp / 'config.json', use_diff=False)
            torch.save(model.state_dict(), tmp / 'base.pth')
            rows = [dict(row, answer=OPTIONS[i]) for i, row in enumerate(self.rows)]
            (tmp / 'train.json').write_text(json.dumps(rows))
            def run(*args, success=True):
                result = subprocess.run([sys.executable, str(ROOT / 'jev.py'), *map(str, args)],
                                        cwd=ROOT, capture_output=True, text=True, timeout=90)
                if success:
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result.stdout
            ckpt = tmp / 'checkpoint'
            run('train', '--weights', tmp / 'base.pth', '--config', tmp / 'config.json',
                '--data', tmp / 'train.json', '--output', ckpt, '--device', 'cpu', '--epochs', '2')
            before = json.loads(run('evaluate', '--checkpoint', ckpt, '--data', tmp / 'train.json', '--device', 'cpu'))
            report = json.loads(run('calibrate', '--checkpoint', ckpt, '--data', tmp / 'train.json', '--device', 'cpu'))
            self.assertLessEqual(report['after']['nll'], before['nll'] + 1e-6)
            predictions = run('predict', '--checkpoint', ckpt, '--data', ROOT / 'data/input.json', '--device', 'cpu')
            self.assertEqual(len(predictions.splitlines()), 2)
            for line in predictions.splitlines():
                self.assertAlmostEqual(sum(json.loads(line)['probabilities'].values()), 1, places=6)
            run('train', '--checkpoint', ckpt, '--data', tmp / 'train.json', '--output', ckpt,
                '--device', 'cpu', success=False)
            temperature_path = ckpt / 'temperature.json'
            payload = json.loads(temperature_path.read_text())
            payload['model_sha256'] = 'stale'
            temperature_path.write_text(json.dumps(payload))
            run('predict', '--checkpoint', ckpt, '--data', ROOT / 'data/input.json', '--device', 'cpu', success=False)


if __name__ == '__main__':
    unittest.main()
