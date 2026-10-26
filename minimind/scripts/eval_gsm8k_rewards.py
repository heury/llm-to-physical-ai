"""Re-score saved GSM8K generations with the exact GRPO scorer.

Run from minimind: .venv/bin/python scripts/eval_gsm8k_rewards.py INPUT --output OUTPUT
INPUT is a JSON list of {response, answer} or {response, gold} records.
`answer` is the full dataset answer; `gold` is a numeric string.
"""
import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.gsm8k_reward import score_gsm8k_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.input).read_text())
    if not rows:
        raise ValueError('No generations to score')
    counts = Counter()
    results = []
    for row in rows:
        answer = row['answer'] if 'answer' in row else '#### ' + format(Decimal(str(row['gold'])), 'f')
        reward, reason = score_gsm8k_response(row['response'], answer)
        counts[reason] += 1
        results.append(dict(row, reward=reward, reward_reason=reason))
    summary = dict(total=len(rows), counts=dict(counts), mean_reward=counts['correct']/len(rows))
    Path(args.output).write_text(json.dumps(dict(summary=summary, rows=results), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
