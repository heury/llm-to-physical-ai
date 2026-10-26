import unittest

from trainer.gsm8k_reward import calculate_gsm8k_rewards, gsm8k_reward


class GSM8KRewardTests(unittest.TestCase):
    def test_correct_and_incorrect(self):
        gold = '48 + 24 = 72.\n#### 72'
        self.assertEqual(gsm8k_reward('48 + 24 = 72\n#### 72', gold), 1.0)
        self.assertEqual(gsm8k_reward('#### 73', gold), 0.0)

    def test_equivalent_numbers(self):
        for response, gold in [('1,000', '1000.00'), ('+.5', '0.50'),
                               ('-2.50', '-2.5'), ('-0', '0')]:
            with self.subTest(response=response):
                self.assertEqual(gsm8k_reward(f'#### {response}', f'#### {gold}'), 1.0)

    def test_exact_decimal_comparison(self):
        self.assertEqual(gsm8k_reward('#### 9007199254740993', '#### 9007199254740992'), 0.0)
        self.assertEqual(gsm8k_reward('#### 0.30000000000000001', '#### 0.3'), 0.0)

    def test_invalid_or_ambiguous_responses(self):
        for response in ['', 'The answer is 72', '72', '#### 72\n#### 73',
                         '#### 73\n#### 72', '#### 72\nActually 73',
                         '#### 72 dollars', '#### 72/1', '#### 7,2',
                         '#### 72%', '#### 7.2e1', '#### NaN', '#### Infinity']:
            with self.subTest(response=response):
                self.assertEqual(gsm8k_reward(response, '#### 72'), 0.0)

    def test_rationale_is_not_scored(self):
        self.assertEqual(gsm8k_reward('Incorrect reasoning\n#### 72\n', '#### 72'), 1.0)

    def test_bad_gold_raises(self):
        for gold in ['72', '#### NaN', '#### 72\n#### 73']:
            with self.subTest(gold=gold), self.assertRaises(ValueError):
                gsm8k_reward('#### 72', gold)

    def test_prompt_major_alignment(self):
        self.assertEqual(calculate_gsm8k_rewards(
            ['#### 1', '#### 2', '#### 1', '#### 2'], ['#### 1', '#### 2'], 2,
        ), [1.0, 0.0, 0.0, 1.0])

    def test_batch_validation(self):
        self.assertEqual(calculate_gsm8k_rewards([], [], 2), [])
        for count in [0, -1, True, 1.5]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                calculate_gsm8k_rewards([], [], count)
        with self.assertRaises(ValueError):
            calculate_gsm8k_rewards(['#### 1'], ['#### 1'], 2)
        with self.assertRaises(TypeError):
            calculate_gsm8k_rewards('#### 1', ['#### 1'])


if __name__ == '__main__':
    unittest.main()
