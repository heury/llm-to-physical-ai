"""Compare explicit instruction paraphrases and reversed choice order without training."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import DecisionCollator, DecisionDataset, answer_values
from model.decision import load_checkpoint
from scripts.runpod_typed.train import metric_summary


def variant_rows(rows, variant, paraphrases):
    result = copy.deepcopy(rows)
    for row in result:
        for q in row['questions']:
            if variant == 'paraphrase':
                instruction, separator, rubric = q['question'].partition('\nRubric:')
                replacement = paraphrases.get(instruction)
                if not isinstance(replacement, str) or not replacement.strip() or replacement == instruction:
                    raise ValueError(f'Missing or unchanged paraphrase: {instruction!r}')
                q['question'] = replacement + separator + rubric
            elif variant == 'reverse_choices':
                if q['type'] == 'choice':
                    q['choices'].reverse()
                    if 'target_probabilities' in q:
                        q['target_probabilities'].reverse()
            elif variant != 'original':
                raise ValueError(f'Unknown variant: {variant}')
    return result


def predict(model, tokenizer, max_length, rows, batch_size, device):
    collate = DecisionCollator(tokenizer, max_length)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            batch = {k: v.to(device) for k, v in collate(selected).items()}
            batch.pop('labels')
            probabilities = model(**batch)['probabilities'].cpu().tolist()
            counts = [len(answer_values(q)) for r in selected for q in r['questions']]
            predictions.extend(p[:count] for p, count in zip(probabilities, counts))
    return predictions


def compare(rows, variants, baseline, predictions):
    """Align by JSON answer value before measuring sensitivity."""
    items, changes, variations, choice_changes = [], [], [], []
    cursor = 0
    for row, variant in zip(rows, variants):
        for q, changed in zip(row['questions'], variant['questions']):
            values, new_values = answer_values(q), answer_values(changed)
            p = [predictions[cursor][new_values.index(value)] for value in values]
            original = baseline[cursor]
            target = q.get('target_probabilities')
            if target is None:
                target = [float(value == q['answer']) for value in values]
            if (len(target) != len(values) or any(not 0 <= v <= 1 for v in target)
                    or abs(sum(target) - 1) > 1e-4):
                raise ValueError(f"Invalid target_probabilities for {q['id']}")
            raw = predictions[cursor]
            items.append(dict(type=q['type'], workflow=row.get('workflow', 'unspecified'),
                              p=raw, g=[target[values.index(v)] for v in new_values]))
            # Preserve decoder first-argmax behavior even when probabilities tie.
            selected = new_values[max(range(len(raw)), key=raw.__getitem__)]
            previous = values[max(range(len(original)), key=original.__getitem__)]
            changed_answer = int(selected != previous)
            changes.append(changed_answer)
            variations.append(sum(abs(a-b) for a, b in zip(p, original)) / 2)
            if q['type'] == 'choice':
                choice_changes.append(changed_answer)
            cursor += 1
    return {
        'metrics': metric_summary(items),
        'answer_change_rate': sum(changes) / len(changes),
        'mean_total_variation': sum(variations) / len(variations),
        'choice_questions': len(choice_changes),
        'choice_answer_change_rate': (sum(choice_changes) / len(choice_changes)
                                      if choice_changes else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True, help='Labelled held-out JSONL')
    parser.add_argument('--paraphrases', default=str(Path(__file__).with_name('typed_paraphrases.json')))
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('batch-size and threads must be positive')
    torch.set_num_threads(args.threads)
    rows = DecisionDataset(args.data).rows
    paraphrases = json.loads(Path(args.paraphrases).read_text())
    variants = {name: variant_rows(rows, name, paraphrases)
                for name in ('original', 'paraphrase', 'reverse_choices')}
    model, tokenizer, max_length = load_checkpoint(args.checkpoint, args.device)
    baseline = predict(model, tokenizer, max_length, rows, args.batch_size, args.device)
    report = {'arguments': vars(args), 'records': len(rows), 'paraphrases': paraphrases,
              'note': 'Fixed instruction paraphrases and reversed choice lists; not unseen-schema evaluation.',
              'variants': {}}
    for name, changed in variants.items():
        predictions = baseline if name == 'original' else predict(
            model, tokenizer, max_length, changed, args.batch_size, args.device)
        report['variants'][name] = compare(rows, changed, baseline, predictions)
        print(json.dumps({name: report['variants'][name]}), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
