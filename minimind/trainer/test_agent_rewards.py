import json
import unittest
from unittest.mock import Mock
from trainer.train_agent import TOOLS, calculate_rewards, validate_gt_in_text

class ToolRewardTests(unittest.TestCase):
    def score(self, turns, gt=None, names=None, unfinished=False, reward_model=None):
        names = names or ['calculate_math']
        tools = [t for t in TOOLS if t['function']['name'] in names]
        return calculate_rewards(['<|im_start|>user\nCalculate 37*19.<|im_end|>'],
            [turns[-1]], [gt if gt is not None else ['703']], [tools], 1,
            reward_model, device='cpu', turn_outputs_batch=[turns],
            unfinished_batch=[unfinished]).item()

    def call(self, expression='37*19'):
        return '<tool_call>' + json.dumps({'name':'calculate_math', 'arguments':{'expression':expression}}) + '</tool_call>'

    def test_refusal_cannot_receive_generic_reward(self):
        rm = Mock(); rm.get_score.return_value = 3.0
        self.assertEqual(self.score(['The query cannot be answered with the provided tools.'], reward_model=rm), -1)
        rm.get_score.assert_not_called()

    def test_correct_answer_without_required_tool_fails(self):
        self.assertEqual(self.score(['The result is 703.']), -1)

    def test_success_beats_refusal_and_malformed_attempt(self):
        self.assertEqual(self.score([self.call(), 'The result is 703.']), 3)
        self.assertEqual(self.score(['<tool_call>[]</tool_call>']), -1.5)
        self.assertEqual(self.score(['<tool_call>{']), -1.5)

    def test_failed_execution_is_not_success(self):
        self.assertEqual(self.score([self.call('1/0'), 'The result is 703.']), -1.5)

    def test_wrong_tool_result_cannot_be_hidden_by_correct_final_text(self):
        self.assertEqual(self.score([self.call('1+1'), 'The result is 703.']), -0.5)

    def test_correct_execution_requires_correct_final_text(self):
        self.assertEqual(self.score([self.call(), 'The result is 700.']), -0.5)

    def test_unknown_tool_or_bad_argument_type(self):
        for call in [{'name':'unknown', 'arguments':{}}, {'name':'calculate_math', 'arguments':{'expression':42}}, {'name':'calculate_math', 'arguments':'[]'}]:
            self.assertEqual(self.score(['<tool_call>'+json.dumps(call)+'</tool_call>', '703']), -1.5)

    def test_unfinished_trajectory_is_not_success(self):
        self.assertEqual(self.score([self.call(), '703'], unfinished=True), -2.0)

    def test_repetition_is_not_better_than_refusal(self):
        self.assertLess(self.score([self.call(), self.call(), self.call()], unfinished=True),
                        self.score(['Cannot answer.']))
        self.assertLess(self.score([self.call(), self.call(), '703']), 3.0)

    def test_numeric_match_does_not_accept_substrings(self):
        self.assertEqual(validate_gt_in_text('1703 or 7030 or 703.5', ['703']), set())
        self.assertEqual(validate_gt_in_text('The result is 1,703.', ['1703']), {'1703'})

    def test_general_dialogue_keeps_reward_model(self):
        rm = Mock(); rm.get_score.return_value = 0.25
        self.assertEqual(self.score(['A helpful answer.'], gt=[], reward_model=rm), 0.75)
        rm.get_score.assert_called_once()

    def test_one_call_can_return_multiple_ground_truth_fields(self):
        call = '<tool_call>{"name":"get_current_weather","arguments":{"location":"Beijing"}}</tool_call>'
        self.assertEqual(self.score([call, '28°C and Clear'], gt=['28°C','Clear'], names=['get_current_weather']), 3)

if __name__ == '__main__': unittest.main()
