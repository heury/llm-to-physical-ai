import copy
import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from dataset import ByteTokenizer, DecisionCollator, DecisionDataset, validate_row
from model.decision import DecisionModel, load_checkpoint, save_checkpoint
from model.model_minimind import MiniMindConfig
from train import evaluate


def rows():
    return [
        {"input": "parcel late", "questions": [
            {"id": "route", "type": "choice", "question": "Which team?", "choices": ["shipping", "account"], "answer": "shipping"},
            {"id": "urgent", "type": "score", "question": "Urgency?", "min": 1, "max": 3, "answer": 3},
            {"id": "cancel", "type": "noul", "question": "Cancel requested?", "answer": False},
        ]},
        {"input": "cancel it now", "questions": [
            {"id": "cancel", "type": "noul", "question": "Cancel requested?", "answer": True},
        ]},
    ]


@pytest.fixture
def setup():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    config = MiniMindConfig(hidden_size=32, num_hidden_layers=2, vocab_size=260,
                           num_attention_heads=4, num_key_value_heads=2,
                           intermediate_size=64, max_position_embeddings=512)
    return DecisionModel(config), DecisionCollator(ByteTokenizer(), 512), ByteTokenizer()


def test_one_backbone_call_with_one_sequence_per_record(setup):
    model, collate, tokenizer = setup
    seen = []
    handle = model.backbone.register_forward_pre_hook(lambda _, args, kwargs: seen.append(kwargs["input_ids"].shape), with_kwargs=True)
    batch = collate(rows())
    out = model(**batch)
    handle.remove()
    assert seen == [batch["input_ids"].shape]
    assert batch["input_ids"].shape[0] == 2
    prefix = [tokenizer.bos_token_id] + tokenizer.encode(rows()[0]["input"]) + [tokenizer.sep_token_id]
    assert batch["input_ids"][0, :len(prefix)].tolist() == prefix
    assert out["logits"].shape == (4, 3)
    assert out["probabilities"][0, 2] == 0
    torch.testing.assert_close(out["probabilities"].sum(-1), torch.ones(4))
    out["loss"].backward()
    assert model.lm_head.weight.grad.abs().sum() > 0
    assert model.backbone.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert not hasattr(model, "head")


def test_questions_are_independent_and_order_invariant(setup):
    model, collate, _ = setup
    model.eval()
    first = rows()[0]
    a = model(**collate([first]))["probabilities"]
    reordered = copy.deepcopy(first)
    reordered["questions"].reverse()
    b = model(**collate([reordered]))["probabilities"]
    torch.testing.assert_close(a, b.flip(0), atol=1e-6, rtol=1e-5)
    changed = copy.deepcopy(first)
    changed["questions"][0]["question"] = "Something completely different?"
    c = model(**collate([changed]))["probabilities"]
    torch.testing.assert_close(a[1:], c[1:], atol=1e-6, rtol=1e-5)
    single = copy.deepcopy(first)
    single["questions"] = [single["questions"][1]]
    d = model(**collate([single]))["probabilities"][0]
    torch.testing.assert_close(a[1], d, atol=1e-6, rtol=1e-5)


def test_batch_padding_and_attention_paths(setup):
    model, collate, _ = setup
    model.eval()
    a = model(**collate(rows()[1:]))["probabilities"][0, :2]
    batch = collate(rows())
    b = model(**batch)["probabilities"][-1, :2]
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    expected = model(**batch)["logits"]
    for layer in model.backbone.layers:
        layer.self_attn.flash = False
    torch.testing.assert_close(model(**batch)["logits"], expected, atol=1e-5, rtol=1e-5)


def test_labels_are_not_in_prompt_and_partial_labels_work(setup):
    model, collate, _ = setup
    original = rows()[0]
    changed = copy.deepcopy(original)
    changed["questions"][0]["answer"] = "account"
    assert torch.equal(collate([original])["input_ids"], collate([changed])["input_ids"])
    del changed["questions"][1]["answer"]
    assert collate([changed])["labels"].tolist() == [1, -100, 0]
    assert torch.isfinite(model(**collate([changed]))["loss"])
    for question in changed["questions"]:
        question.pop("answer", None)
    assert model(**collate([changed]))["loss"] is None


def test_learns_typed_answers_and_roundtrips(setup, tmp_path):
    model, collate, tokenizer = setup
    batch = collate(rows())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial = model(**batch)["loss"].item()
    for _ in range(100):
        optimizer.zero_grad()
        loss = model(**batch)["loss"]
        loss.backward()
        optimizer.step()
    model.eval()
    out = model(**batch)
    assert out["loss"].item() < initial * 0.1
    decoded = model.decode(rows(), out["probabilities"])
    assert decoded[0]["answers"] == {"route": "shipping", "urgent": 3, "cancel": False}
    assert decoded[1]["answers"] == {"cancel": True}
    save_checkpoint(tmp_path, model, tokenizer, 512)
    restored, tok, length = load_checkpoint(tmp_path)
    torch.testing.assert_close(restored(**batch)["logits"], out["logits"])
    assert tok.encode("안녕") == tokenizer.encode("안녕") and length == 512


def test_lm_conversion_and_head_only_training(setup):
    model, collate, _ = setup
    state = {"model." + k: v.clone() for k, v in model.backbone.state_dict().items()}
    state["lm_head.weight"] = model.lm_head.weight.detach().clone()
    target = DecisionModel(model.config)
    target.load_lm_state_dict(state)
    torch.testing.assert_close(target.lm_head.weight, state["lm_head.weight"])
    assert target.lm_head.weight is target.backbone.embed_tokens.weight
    original = {k: v.clone() for k, v in target.backbone.state_dict().items()}
    target.freeze_backbone()
    assert target.lm_head.weight is not target.backbone.embed_tokens.weight
    optimizer = torch.optim.AdamW((p for p in target.parameters() if p.requires_grad), lr=0.01)
    target(**collate(rows()))["loss"].backward()
    optimizer.step()
    assert all(torch.equal(v, original[k]) for k, v in target.backbone.state_dict().items())
    assert target.lm_head.weight.grad is not None
    del state["model.norm.weight"]
    with pytest.raises(RuntimeError):
        target.load_lm_state_dict(state)


@pytest.mark.parametrize("change", [
    {"type": "other"}, {"choices": []}, {"choices": ["a", "a"]},
    {"answer": "missing"}, {"question": ""}, {"id": ""},
])
def test_rejects_bad_choice_questions(change):
    row = rows()[0]
    row["questions"][0].update(change)
    with pytest.raises(ValueError):
        validate_row(row)


@pytest.mark.parametrize("answer", [True, "2", 0, 4, 1.5, float("nan")])
def test_rejects_bad_score(answer):
    row = rows()[0]
    row["questions"][1]["answer"] = answer
    with pytest.raises(ValueError):
        validate_row(row)


@pytest.mark.parametrize("answer", [0, 1, "true", None])
def test_noul_requires_real_boolean(answer):
    row = rows()[0]
    row["questions"][2]["answer"] = answer
    with pytest.raises(ValueError):
        validate_row(row)


def test_schema_dataset_and_legacy_rows(setup, tmp_path):
    model, collate, _ = setup
    old = {"input": "hello", "choices": ["a", "b"], "label": 1}
    assert torch.isfinite(model(**collate([old]))["loss"])
    row = rows()[0]
    row["questions"][1]["id"] = "route"
    with pytest.raises(ValueError, match="id"):
        validate_row(row)
    with pytest.raises(ValueError, match="max_length"):
        DecisionCollator(ByteTokenizer(), 8)(rows())
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"input": "x", "questions": []}))
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        DecisionDataset(path)


def test_checkpoint_v1_rejected_clearly(tmp_path):
    (tmp_path / "decision_config.json").write_text(json.dumps({"format_version": 1}))
    with pytest.raises(ValueError, match="retrain"):
        load_checkpoint(tmp_path)


def test_hf_tokenizer_roundtrip(setup, tmp_path):
    model, _, _ = setup
    vocab = {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3,
             **{chr(65 + i): 4 + i for i in range(26)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                       pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]")
    config = model.config.to_dict()
    config["vocab_size"] = len(tokenizer)
    model = DecisionModel(MiniMindConfig(**config)).eval()
    batch = DecisionCollator(tokenizer, 512)(rows())
    expected = model(**batch)["logits"]
    save_checkpoint(tmp_path, model, tokenizer, 512)
    restored, tok, _ = load_checkpoint(tmp_path)
    assert tok.encode("A", add_special_tokens=False) == [4]
    torch.testing.assert_close(restored(**batch)["logits"], expected)


def test_one_question_matches_ordinary_causal_lm(setup):
    model, collate, _ = setup
    model.eval()
    batch = collate(rows()[1:])
    expected = model.backbone(batch["input_ids"])
    packed = model.backbone(batch["input_ids"], attention_mask=batch["attention_mask"], position_ids=batch["position_ids"])
    torch.testing.assert_close(packed, expected, atol=1e-6, rtol=1e-5)
    full_vocab = model.lm_head(expected[:, -1])
    torch.testing.assert_close(model(**batch)["logits"], full_vocab.gather(1, batch["answer_token_ids"]), atol=1e-5, rtol=1e-5)


def test_untied_frozen_head_roundtrip(setup, tmp_path):
    model, collate, tokenizer = setup
    model.freeze_backbone()
    optimizer = torch.optim.AdamW([model.lm_head.weight], lr=0.01)
    batch = collate(rows())
    model(**batch)["loss"].backward()
    optimizer.step()
    model.eval()
    assert not torch.equal(model.lm_head.weight, model.backbone.embed_tokens.weight)
    save_checkpoint(tmp_path, model, tokenizer, 512)
    restored, _, _ = load_checkpoint(tmp_path)
    assert not restored.config.tie_word_embeddings
    assert restored.lm_head.weight is not restored.backbone.embed_tokens.weight
    torch.testing.assert_close(restored(**batch)["logits"], model(**batch)["logits"])


@pytest.mark.parametrize("change", [{"min": True}, {"min": 4, "max": 3}, {"min": 0, "max": 26}])
def test_score_range_validation(change):
    row = rows()[0]
    row["questions"][1].update(change)
    with pytest.raises(ValueError):
        validate_row(row)


def test_missing_labels_and_too_many_options():
    row = rows()[0]
    del row["questions"][0]["answer"]
    with pytest.raises(ValueError, match="answer"):
        validate_row(row)
    row["questions"][0]["choices"] = [str(i) for i in range(27)]
    with pytest.raises(ValueError, match="26"):
        validate_row(row, require_label=False)


def test_custom_rope_settings_survive_checkpoint(setup, tmp_path):
    model, collate, tokenizer = setup
    config = model.config.to_dict()
    scaling = dict(type="yarn", factor=3, original_max_position_embeddings=32,
                   beta_fast=8, beta_slow=1, attention_factor=1.2)
    config.update(inference_rope_scaling=True, rope_scaling=scaling)
    model = DecisionModel(MiniMindConfig(**config)).eval()
    assert model.config.rope_scaling == scaling
    batch = collate(rows())
    expected = model(**batch)["logits"]
    save_checkpoint(tmp_path, model, tokenizer, 512)
    restored, _, _ = load_checkpoint(tmp_path)
    assert restored.config.rope_scaling == scaling
    torch.testing.assert_close(restored(**batch)["logits"], expected, atol=0, rtol=0)


def test_evaluation_skips_unlabelled_batches(setup):
    model, collate, _ = setup
    labelled = rows()
    unlabelled = copy.deepcopy(labelled)
    for row in unlabelled:
        for question in row["questions"]:
            del question["answer"]
    expected = evaluate(model, [collate(labelled)], "cpu")
    actual = evaluate(model, [collate(unlabelled), collate(labelled)], "cpu")
    assert actual == expected
    with pytest.raises(ValueError, match="labelled"):
        evaluate(model, [collate(unlabelled)], "cpu")


@pytest.mark.parametrize("change", [
    {"head_dim": 0}, {"head_dim": -2}, {"num_attention_heads": 0},
    {"max_position_embeddings": 0}, {"intermediate_size": 0},
])
def test_invalid_dimensions_fail_clearly(setup, change):
    config = setup[0].config.to_dict()
    config.update(change)
    with pytest.raises(ValueError):
        DecisionModel(MiniMindConfig(**config))
