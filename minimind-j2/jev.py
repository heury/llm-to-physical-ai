"""MiniMind Jev: frozen backbone, choice-token LM head, temperature scaling."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

ROOT = Path(__file__).resolve().parent
OPTIONS = tuple('ABCDE')


def validate(item, labeled=False):
    if not isinstance(item, dict):
        raise ValueError('Each example must be an object')
    for key in ('question', *OPTIONS):
        if not isinstance(item.get(key), str) or not item[key].strip():
            raise ValueError(f'{key} must be a nonempty string')
    if labeled and item.get('answer') not in OPTIONS:
        raise ValueError('Labeled examples require answer A, B, C, D, or E')
    return item


def read_data(path, labeled=False, limit=None):
    path = Path(path)
    if limit is not None and limit <= 0:
        raise ValueError('limit must be positive')
    if path.suffix == '.jsonl':
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('Input must contain a nonempty list of examples')
    rows = rows[:limit]
    return [validate(row, labeled) for row in rows]


def prompt(tokenizer, row):
    validate(row)
    text = row['question'] + '\n' + '\n'.join(f'{key}. {row[key]}' for key in OPTIONS) + '\nAnswer:'
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': text}], tokenize=False,
        add_generation_prompt=True, open_thinking=False, enable_thinking=False,
    )


def option_ids(tokenizer):
    encoded = [tokenizer.encode(key, add_special_tokens=False) for key in OPTIONS]
    if any(len(ids) != 1 for ids in encoded) or len({ids[0] for ids in encoded}) != 5:
        raise ValueError('A–E must map to five distinct single tokens')
    return [ids[0] for ids in encoded]


def freeze_backbone(model):
    # Clone even when tied: updating the head must never mutate input embeddings.
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
    model.config.tie_word_embeddings = False
    model.requires_grad_(False)
    model.lm_head.requires_grad_(True)
    model.eval()  # Frozen dropout and MoE routing must remain deterministic.


class Jev:
    def __init__(self, model, tokenizer, max_length=2048):
        self.model, self.tokenizer = model.eval(), tokenizer
        self.ids = option_ids(tokenizer)
        if max_length <= 0 or max_length > model.config.max_position_embeddings:
            raise ValueError('max_length must be within the model position limit')
        if max(self.ids) >= model.config.vocab_size or len(tokenizer) > model.config.vocab_size:
            raise ValueError('Tokenizer vocabulary does not fit the model')
        self.max_length = max_length

    def logits(self, rows):
        if not rows:
            raise ValueError('Empty batch')
        seqs = [self.tokenizer.encode(prompt(self.tokenizer, row), add_special_tokens=False) for row in rows]
        if any(not seq or len(seq) > self.max_length for seq in seqs):
            raise ValueError('Prompt exceeds max_length; shorten it explicitly (no silent truncation)')
        device = next(self.model.parameters()).device
        lengths = torch.tensor([len(seq) for seq in seqs], device=device)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise ValueError('Tokenizer needs a pad or EOS token')
        # Right padding preserves original RoPE positions and avoids all-masked queries.
        ids = torch.full((len(rows), int(lengths.max())), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(seqs):
            ids[i, :len(seq)] = torch.tensor(seq, device=device)
            mask[i, :len(seq)] = 1
        with torch.no_grad():
            hidden, _, _ = self.model.model(ids, attention_mask=mask)
            last = hidden[torch.arange(len(rows), device=device), lengths - 1]
        # Equivalent to lm_head(last)[:, choice_ids], without full-vocabulary logits.
        return F.linear(last, self.model.lm_head.weight[self.ids]).float()

    def collect(self, rows, batch_size):
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        with torch.no_grad():
            return torch.cat([self.logits(rows[i:i + batch_size]).cpu()
                              for i in range(0, len(rows), batch_size)])


def labels_for(rows):
    return torch.tensor([OPTIONS.index(validate(row, True)['answer']) for row in rows])


def check_temperature(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Temperature must be finite and positive')
    return value


def metrics(logits, labels, temperature=1.0):
    scaled = logits / check_temperature(temperature)
    probs = scaled.softmax(-1)
    confidence, predictions = probs.max(-1)
    correct = predictions.eq(labels).float()
    ece = 0.0
    for low, high in zip(torch.linspace(0, 1, 16)[:-1], torch.linspace(0, 1, 16)[1:]):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            ece += float(inside.float().mean() * (confidence[inside].mean() - correct[inside].mean()).abs())
    return {'count': len(labels), 'accuracy': float(correct.mean()),
            'nll': float(F.cross_entropy(scaled, labels)), 'ece': ece,
            'brier': float(((probs - F.one_hot(labels, 5)) ** 2).sum(-1).mean()),
            'temperature': float(temperature)}


def fit_temperature(logits, labels):
    # Optimize inverse temperature on a bounded interval. NLL is convex in 1/T.
    # Include T=1 so calibration never worsens the fitting-set NLL.
    low, high = 0.001, 100.0
    for _ in range(80):
        left, right = low + (high - low) / 3, high - (high - low) / 3
        if F.cross_entropy(logits * left, labels) < F.cross_entropy(logits * right, labels):
            high = right
        else:
            low = left
    candidates = [1.0, 1 / ((low + high) / 2)]
    return min(candidates, key=lambda t: float(F.cross_entropy(logits / t, labels)))


def load_jev(args):
    if args.checkpoint:
        folder = Path(args.checkpoint)
        config = MiniMindConfig(**json.loads((folder / 'config.json').read_text()))
        model = MiniMindForCausalLM(config)
        model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True), strict=True)
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        meta = json.loads((folder / 'jev.json').read_text())
        max_length = args.max_length or meta['max_length']
    else:
        if not args.weights or not args.config:
            raise ValueError('Provide --checkpoint, or both --weights and --config')
        config = MiniMindConfig(**json.loads(Path(args.config).read_text()))
        model = MiniMindForCausalLM(config)
        model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True), strict=True)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        max_length = args.max_length or min(2048, config.max_position_embeddings)
        meta = None
    runner = Jev(model.to(args.device), tokenizer, max_length)
    if meta is not None and (meta['options'] != list(OPTIONS) or meta['option_token_ids'] != runner.ids):
        raise ValueError('Checkpoint option metadata does not match tokenizer')
    return runner


def save_checkpoint(runner, output, training):
    folder = Path(output)
    # Avoid silently reusing stale calibration or replacing an existing model.
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Output directory must be empty')
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(runner.model.state_dict(), folder / 'model.pt')
    runner.model.config.to_json_file(folder / 'config.json', use_diff=False)
    runner.tokenizer.save_pretrained(folder)
    (folder / 'jev.json').write_text(json.dumps({
        'format_version': 1, 'options': list(OPTIONS), 'option_token_ids': runner.ids,
        'max_length': runner.max_length, 'training': training,
    }, indent=2) + '\n')


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stored_temperature(args):
    if args.temperature is not None:
        return check_temperature(args.temperature)
    if args.checkpoint and (Path(args.checkpoint) / 'temperature.json').exists():
        folder = Path(args.checkpoint)
        calibration = json.loads((folder / 'temperature.json').read_text())
        if calibration['model_sha256'] != fingerprint(folder / 'model.pt'):
            raise ValueError('Temperature belongs to different model weights')
        return check_temperature(calibration['temperature'])
    return 1.0


def train(args, runner):
    if args.epochs <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('epochs and learning_rate must be positive')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    rows = read_data(args.data, True, args.limit)
    freeze_backbone(runner.model)
    optimizer = torch.optim.Adam(runner.model.lm_head.parameters(), lr=args.learning_rate)
    rng = random.Random(args.seed)
    losses = []
    for epoch in range(args.epochs):
        rng.shuffle(rows)
        total = 0.0
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            logits = runner.logits(batch)
            loss = F.cross_entropy(logits, labels_for(batch).to(logits.device))
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        losses.append(total / len(rows))
        print(json.dumps({'epoch': epoch + 1, 'train_nll': losses[-1]}))
    save_checkpoint(runner, args.output, {
        'seed': args.seed, 'epochs': args.epochs, 'learning_rate': args.learning_rate,
        'batch_size': args.batch_size, 'examples': len(rows), 'train_nll': losses,
        'data_sha256': fingerprint(args.data),
    })


def prepare(args):
    from datasets import load_dataset
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    splits = load_dataset('tau/commonsense_qa')
    def convert(example):
        row = {'question': example['question']}
        row.update(zip(example['choices']['label'], example['choices']['text']))
        row['answer'] = example['answerKey']
        return validate(row, True)
    train_rows = [convert(row) for row in splits['train']]
    validation = [convert(row) for row in splits['validation']]
    random.Random(args.seed).shuffle(validation)
    middle = len(validation) // 2
    output.mkdir(parents=True, exist_ok=True)
    # Official test labels are unavailable; reserve disjoint validation subsets.
    for name, rows in [('train', train_rows), ('calibration', validation[:middle]), ('eval', validation[middle:])]:
        (output / f'{name}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        print(f'{name}: {len(rows)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Download CommonsenseQA and create disjoint calibration/eval sets')
    prep.add_argument('--output', default='dataset')
    prep.add_argument('--seed', type=int, default=0)
    for name in ('train', 'predict', 'evaluate', 'calibrate'):
        p = sub.add_parser(name)
        source = p.add_mutually_exclusive_group(required=True)
        source.add_argument('--checkpoint')
        source.add_argument('--weights', help='Native MiniMind state dict (.pth)')
        p.add_argument('--config', help='JSON config matching native weights')
        p.add_argument('--tokenizer', default=str(ROOT / 'model'))
        p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        p.add_argument('--max-length', type=int)
        p.add_argument('--batch-size', type=int, default=16)
        p.add_argument('--limit', type=int)
        p.add_argument('--data', required=True)
        if name == 'train':
            p.add_argument('--output', required=True)
            p.add_argument('--epochs', type=int, default=1)
            p.add_argument('--learning-rate', type=float, default=1e-4)
            p.add_argument('--seed', type=int, default=0)
        if name in ('predict', 'evaluate'):
            p.add_argument('--temperature', type=float, help='Override stored calibration (1 = uncalibrated)')
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            prepare(args)
            return
        if args.batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if args.command == 'calibrate' and not args.checkpoint:
            raise ValueError('Calibration requires --checkpoint')
        torch.manual_seed(getattr(args, 'seed', 0))
        runner = load_jev(args)
        if args.command == 'train':
            train(args, runner)
            return
        rows = read_data(args.data, args.command != 'predict', args.limit)
        logits = runner.collect(rows, args.batch_size)
        if args.command == 'calibrate':
            labels = labels_for(rows)
            temperature = fit_temperature(logits, labels)
            report = {'temperature': temperature, 'model_sha256': fingerprint(Path(args.checkpoint) / 'model.pt'),
                      'data_sha256': fingerprint(args.data),
                      'before': metrics(logits, labels), 'after': metrics(logits, labels, temperature)}
            (Path(args.checkpoint) / 'temperature.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2))
        elif args.command == 'evaluate':
            print(json.dumps(metrics(logits, labels_for(rows), stored_temperature(args)), indent=2))
        else:
            probs = (logits / stored_temperature(args)).softmax(-1)
            for row, values in zip(rows, probs.tolist()):
                answer = OPTIONS[max(range(5), key=values.__getitem__)]
                print(json.dumps({'question': row['question'], 'answer': answer, 'text': row[answer],
                                  'probabilities': dict(zip(OPTIONS, values))}, ensure_ascii=False))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
