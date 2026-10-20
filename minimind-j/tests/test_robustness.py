import copy
import importlib
import pytest


def module():
    return importlib.import_module('scripts.evaluate_robustness')


def row():
    return {'input': 'x', 'workflow': 'test', 'questions': [
        {'id': 'route', 'type': 'choice', 'question': 'Which?\nRubric:\na: first\nb: second',
         'choices': ['a', 'b'], 'answer': 'a', 'target_probabilities': [.8, .2]},
        {'id': 'ok', 'type': 'noul', 'question': 'OK?', 'answer': True},
    ]}


def test_variants_preserve_semantics_and_do_not_mutate():
    source = row()
    original = copy.deepcopy(source)
    m = module()
    reverse = m.variant_rows([source], 'reverse_choices', {})[0]
    assert reverse['questions'][0]['choices'] == ['b', 'a']
    assert reverse['questions'][0]['target_probabilities'] == [.2, .8]
    assert reverse['questions'][0]['answer'] == 'a'
    assert reverse['questions'][1] == source['questions'][1]
    changed = m.variant_rows([source], 'paraphrase', {'Which?': 'Select one.', 'OK?': 'Is it OK?'})[0]
    assert changed['questions'][0]['question'] == 'Select one.\nRubric:\na: first\nb: second'
    assert source == original


def test_comparison_aligns_by_value_and_handles_mixed_types():
    m = module()
    source = [row()]
    reverse = m.variant_rows(source, 'reverse_choices', {})
    result = m.compare(source, reverse, [[.8, .2], [.1, .9]], [[.2, .8], [.1, .9]])
    assert result['answer_change_rate'] == 0
    assert result['mean_total_variation'] == pytest.approx(0)
    assert result['metrics']['accuracy'] == 1
    assert result['choice_answer_change_rate'] == 0


def test_missing_paraphrase_fails_instead_of_silent_identity():
    with pytest.raises(ValueError, match='paraphrase'):
        module().variant_rows([row()], 'paraphrase', {})


def test_tied_predictions_use_actual_decoded_choice_after_reordering():
    m = module()
    source = [row()]
    reverse = m.variant_rows(source, 'reverse_choices', {})
    result = m.compare(source, reverse, [[.5, .5], [.1, .9]], [[.5, .5], [.1, .9]])
    assert result['choice_answer_change_rate'] == 1
    assert result['answer_change_rate'] == .5
    assert result['mean_total_variation'] == 0
