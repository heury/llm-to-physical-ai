"""Deterministic GSM8K outcome rewards, with no model or torch dependency.

Pass decoded assistant completions only (skip_special_tokens=True), and keep
the dataset's original `answer` column outside the model prompt. Ask the model:
"Solve step by step. End your answer with #### followed by the final number."

For trainer/train_grpo.py's prompt-major rollout order:
    scores = calculate_gsm8k_rewards(completions, batch['answer'], num_generations)
    rewards = torch.tensor(scores, dtype=torch.float32, device=device)

Select --reward_type gsm8k in trainer/train_grpo.py to use the GSM8K loader.
The loader preserves `answer` separately and uses only `question` in prompts.
Use the train split for optimization and reserve the test split for evaluation.
"""

import re
from decimal import Decimal
from typing import Sequence


# ASCII digits; commas must separate groups of three. No expressions, units,
# percentages, scientific notation, NaN, or infinity are accepted.
_NUMBER = re.compile(r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)")


def extract_gsm8k_answer(text: str) -> Decimal | None:
    """Parse a single #### marker followed by a numeric suffix through EOF.

    Rationale before the marker is unrestricted. Reject multiple markers to
    prevent enumeration of candidate answers. Decimal equality avoids floating
    point rounding and treats 1,000, 1000.0 and +1000 as the same value.
    """
    if text.count('####') != 1:
        return None
    value = text.split('####', 1)[1].strip()
    if _NUMBER.fullmatch(value) is None:
        return None
    return Decimal(value.replace(',', ''))


def score_gsm8k_response(response: str, answer: str) -> tuple[float, str]:
    """Shared training/evaluation score and mutually exclusive diagnostic reason.

    Gold uses the dataset suffix convention; generated answers must put ####
    at the beginning of the final line. One optional, closed think block is
    allowed before the final answer. No reward is given for format alone.
    """
    expected = extract_gsm8k_answer(answer)
    if expected is None:
        raise ValueError('GSM8K gold answer must contain one #### numeric suffix')
    final = response
    if '<think>' in response or '</think>' in response:
        if response.count('<think>') != 1 or response.count('</think>') != 1:
            return 0.0, 'format_error'
        opening = response.index('<think>')
        closing = response.index('</think>')
        if opening > closing or '####' in response[:closing]:
            return 0.0, 'format_error'
        final = response[closing + len('</think>'):]
    if re.search(r'(?:^|\n)[ \t]*####[ \t]+[^\n]+\s*\Z', final) is None:
        return 0.0, 'format_error'
    predicted = extract_gsm8k_answer(response)
    if predicted is None:
        return 0.0, 'format_error'
    correct = predicted == expected
    return float(correct), 'correct' if correct else 'wrong_number'


def gsm8k_reward(response: str, answer: str) -> float:
    """Return 1 for a valid, exactly correct final answer; otherwise 0.

    `answer` is the original GSM8K rationale + #### number, not a bare number.
    Malformed gold data raises rather than silently teaching incorrect rewards.
    This checks the final outcome, not the mathematical validity of the rationale.
    """
    return score_gsm8k_response(response, answer)[0]


def calculate_gsm8k_rewards(
    responses: Sequence[str], answers: Sequence[str], num_generations: int = 1,
) -> list[float]:
    """Score B*G responses ordered [q0_g0, q0_g1, ..., q1_g0, ...].

    Supply B original gold answers. Returns B*G floats in the same order;
    no group normalization or KL penalty is applied here (the trainer owns it).
    """
    if isinstance(num_generations, bool) or not isinstance(num_generations, int) or num_generations < 1:
        raise ValueError('num_generations must be a positive integer')
    if isinstance(responses, str) or isinstance(answers, str):
        raise TypeError('responses and answers must be sequences of strings')
    if len(responses) != len(answers) * num_generations:
        raise ValueError('Expected len(responses) == len(answers) * num_generations')
    return [
        gsm8k_reward(response, answers[index // num_generations])
        for index, response in enumerate(responses)
    ]
