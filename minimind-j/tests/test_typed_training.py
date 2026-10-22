import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.runpod_typed import train
from scripts.runpod_typed.train import convert_record, soft_cross_entropy, split_training
from dataset import ByteTokenizer, DecisionCollator
from metrics import metric_summary


def fixture():
    return json.loads((Path(__file__).parent/'fixtures/typed_decision.json').read_text())


def test_conversion_retains_five_questions_rubrics_and_soft_gold():
    row=convert_record(fixture())
    assert len(row['questions'])==5
    assert {q['type'] for q in row['questions']}=={'choice','score','noul'}
    q={q['id']:q for q in row['questions']}
    assert 'Let the agent proceed without interruption.' in q['action']['question']
    assert 'High: destructive' in q['risk']['question']
    assert q['risk']['answer']==0
    assert q['needs_review']['answer'] is True
    assert q['action']['target_probabilities']==pytest.approx([.283333,.433333,.25,.033333],abs=1e-6)
    changed=copy.deepcopy(row)
    changed['questions'][0]['target_probabilities']=[1,0,0,0]
    collate=DecisionCollator(ByteTokenizer(),10000)
    assert torch.equal(collate([row])['input_ids'],collate([changed])['input_ids'])


def test_soft_cross_entropy_handles_padding_and_keeps_gradients():
    logits=torch.tensor([[1.,2.,float('-inf')],[2.,1.,0.]],requires_grad=True)
    targets=torch.tensor([[.2,.8,0.],[.3,.4,.3]])
    mask=torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool)
    loss=soft_cross_entropy(logits,targets,mask)
    expected=(-(targets[0,:2]*logits[0,:2].log_softmax(0)).sum()-(targets[1]*logits[1].log_softmax(0)).sum())/2
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0,2]==0


def test_metrics_match_gold_and_vector_brier_definition():
    items=[{'type':'choice','workflow':'a','question':'q','p':[.2,.8], 'g':[.2,.8]},
           {'type':'noul','workflow':'b','question':'n','p':[.75,.25], 'g':[.8,.2]}]
    result=metric_summary(items)
    assert result['accuracy']==1
    assert result['brier']==pytest.approx(.0025)
    assert result['kl_from_gold']>=0
    assert result['by_type']['choice']['brier']==0
    assert result['decisions']==2


def test_stratified_validation_never_uses_official_test():
    rows=[]
    for workflow in ['a','b']:
        for i in range(10):rows.append({'id':f'{workflow}-{i}','workflow':workflow,'input':f'{workflow}-{i}'})
    train,val=split_training(rows,.2,42)
    assert len(train)==16 and len(val)==4
    assert len({r['workflow'] for r in val})==2
    assert not ({r['id'] for r in train}&{r['id'] for r in val})
    train2,val2=split_training(rows,.2,42)
    assert train==train2 and val==val2


@pytest.mark.parametrize('sizes', [[1, 3, 2], [2, 1]])
def test_accumulated_gradients_match_full_question_batch(sizes):
    torch.manual_seed(9)
    model = torch.nn.Linear(3, 2)
    x = torch.randn(sum(sizes), 3)
    targets = torch.softmax(torch.randn(sum(sizes), 2), -1)
    mask = torch.ones_like(targets, dtype=torch.bool)
    soft_cross_entropy(model(x), targets, mask).backward()
    expected = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    offset = 0
    for size in sizes:
        end = offset + size
        (soft_cross_entropy(model(x[offset:end]), targets[offset:end], mask[offset:end]) * size).backward()
        offset = end
    assert hasattr(train, 'normalize_question_gradients')
    train.normalize_question_gradients(model.parameters(), sum(sizes))
    for parameter, gradient in zip(model.parameters(), expected):
        torch.testing.assert_close(parameter.grad, gradient)


@pytest.mark.parametrize('fraction', [0, 1, -.1, 1.1])
def test_validation_split_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match='fraction'):
        split_training([], fraction, 42)
