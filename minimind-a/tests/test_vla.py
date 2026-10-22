from unittest.mock import patch

import pytest
import torch

from model.model_vla import MiniMindVLA, VLAConfig


def small_config(siglip_dir, **kwargs):
    return VLAConfig(
        vlm_config=dict(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=6400,
            intermediate_size=64,
            max_position_embeddings=128,
            image_hidden_size=24,
            image_token_len=16,
        ),
        vision_model_path=kwargs.pop("vision_model_path", str(siglip_dir)),
        expert_dim=32,
        expert_heads=4,
        expert_layers=1,
        state_dim=2,
        action_dim=2,
        chunk_size=4,
        **kwargs,
    )


def batch():
    return dict(
        input_ids=torch.tensor([[12] * 16 + [1, 30, 31, 2]] * 2),
        attention_mask=torch.ones(2, 20, dtype=torch.bool),
        pixel_values=torch.randn(2, 3, 64, 64),
        state=torch.randn(2, 2),
    )


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(7)
    torch.set_num_threads(1)


def test_flow_endpoints_and_target():
    actions, noise = torch.randn(2, 4, 2), torch.randn(2, 4, 2)
    x, v = MiniMindVLA.flow_inputs(actions, noise, torch.zeros(2))
    torch.testing.assert_close(x, noise)
    torch.testing.assert_close(v, actions - noise)
    x, _ = MiniMindVLA.flow_inputs(actions, noise, torch.ones(2))
    torch.testing.assert_close(x, actions)


def test_loss_and_gradients_reach_vision_language_and_action(siglip_dir):
    model = MiniMindVLA(
        small_config(
            siglip_dir,
        )
    )
    loss = model(**batch(), actions=torch.randn(2, 4, 2))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(
        p.grad is None and not p.requires_grad
        for p in model.vlm.vision_encoder.parameters()
    )
    for parameter in [
        model.vlm.vision_proj.mlp[1].weight,
        model.vlm.model.layers[0].self_attn.q_proj.weight,
        model.action_out.weight,
    ]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


def test_action_padding_does_not_change_valid_loss(siglip_dir):
    model = MiniMindVLA(
        small_config(
            siglip_dir,
        )
    ).eval()
    actions, noise = torch.randn(2, 4, 2), torch.randn(2, 4, 2)
    valid = torch.tensor([[True, True, False, False]] * 2)
    kwargs = dict(**batch(), noise=noise, time=torch.ones(2) * 0.5, action_mask=valid)
    first = model(**kwargs, actions=actions)
    changed = actions.clone()
    changed[:, 2:] = 10000
    torch.testing.assert_close(first, model(**kwargs, actions=changed))
    with pytest.raises(ValueError, match="valid"):
        model(**{**kwargs, "action_mask": torch.zeros_like(valid)}, actions=actions)


def test_numeric_action_mask_uses_boolean_validity(siglip_dir):
    model = MiniMindVLA(small_config(siglip_dir)).eval()
    mask = torch.tensor([[2, 1, 0, 0]] * 2)
    kwargs = dict(
        **batch(),
        actions=torch.randn(2, 4, 2),
        noise=torch.randn(2, 4, 2),
        time=torch.full((2,), 0.5),
    )
    expected = model(**kwargs, action_mask=mask.bool())
    torch.testing.assert_close(model(**kwargs, action_mask=mask), expected)


def test_nonfinite_padding_is_excluded_before_loss_and_backward(siglip_dir):
    model = MiniMindVLA(small_config(siglip_dir)).eval()
    actions = torch.randn(2, 4, 2)
    actions[:, 2:] = float("nan")
    noise = torch.randn_like(actions)
    noise[:, 2:] = float("inf")
    loss = model(
        **batch(),
        actions=actions,
        noise=noise,
        time=torch.full((2,), 0.5),
        action_mask=torch.tensor([[True, True, False, False]] * 2),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_training_context_matches_native_vlm_with_dropout(siglip_dir):
    from model.model_vlm import MiniMindVLM, VLMConfig

    config = small_config(siglip_dir)
    config.vlm_config["dropout"] = 0.25
    model = MiniMindVLA(config).train()
    native = MiniMindVLM(VLMConfig(**config.vlm_config), str(siglip_dir)).train()
    native.load_state_dict(model.vlm.state_dict())
    native.vision_encoder.eval()
    inputs = batch()
    torch.manual_seed(123)
    expected = native(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"].float(),
        pixel_values={"pixel_values": inputs["pixel_values"]},
    ).hidden_states
    torch.manual_seed(123)
    actual = model.vlm(
        inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"]
    )
    torch.testing.assert_close(actual, expected)


def test_sampling_reuses_context_and_restores_training_mode(siglip_dir):
    model = MiniMindVLA(
        small_config(
            siglip_dir,
        )
    ).train()
    b = batch()
    noise = torch.randn(2, 4, 2)
    with patch.object(model, "encode_context", wraps=model.encode_context) as encode:
        result = model.sample_actions(**b, noise=noise, steps=5)
    assert encode.call_count == 1 and result.shape == (2, 4, 2)
    assert torch.isfinite(result).all() and model.training
    torch.testing.assert_close(result, model.sample_actions(**b, noise=noise, steps=5))


def test_cached_context_matches_full_training_loss_and_gradients(siglip_dir):
    model = MiniMindVLA(small_config(siglip_dir, freeze_vlm=True)).train()
    inputs = batch()
    actions, noise = torch.randn(2, 4, 2), torch.randn(2, 4, 2)
    time = torch.tensor([0.2, 0.8])
    mask = torch.tensor([[True, True, False, False], [True] * 4])
    parameters = [
        model.context_proj.weight,
        model.state_proj.weight,
        model.action_out.weight,
    ]
    full_loss = model(
        **inputs, actions=actions, noise=noise, time=time, action_mask=mask
    )
    full_gradients = torch.autograd.grad(full_loss, parameters)

    with torch.no_grad():
        hidden = model.vlm(
            inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"]
        )
    memory, valid = model.project_context(
        hidden, inputs["attention_mask"], inputs["state"]
    )
    cached_loss = model.flow_loss(memory, valid, actions, mask, noise, time)
    cached_gradients = torch.autograd.grad(cached_loss, parameters)
    torch.testing.assert_close(cached_loss, full_loss)
    for cached, full in zip(cached_gradients, full_gradients):
        torch.testing.assert_close(cached, full)


def test_masked_flow_loss_uses_valid_scalar_count_and_fp32(siglip_dir):
    model = MiniMindVLA(small_config(siglip_dir)).eval()
    memory, valid = model.encode_context(**batch())
    actions = torch.full((2, 4, 2), 400.0, dtype=torch.float16)
    mask = torch.tensor([[True, False, False, False], [True, True, False, False]])
    with patch.object(model, "velocity", return_value=torch.zeros_like(actions)):
        loss = model.flow_loss(memory, valid, actions, mask, torch.zeros_like(actions))
    # 모든 유효 원소의 오차는 400². 패딩 비율과 무관하며 FP16에서는 overflow한다.
    torch.testing.assert_close(loss, torch.tensor(160000.0))


@pytest.mark.parametrize("steps", [1, 10])
def test_euler_constant_field_reaches_known_endpoint(siglip_dir, steps):
    model = MiniMindVLA(small_config(siglip_dir)).eval()
    inputs = batch()
    noise, target = torch.randn(2, 4, 2), torch.randn(2, 4, 2)
    original_noise = noise.clone()
    # 일정한 속도장에서는 어떤 step 수로도 정해진 끝점에 도달해야 한다.
    with patch.object(model, "velocity", return_value=target - noise) as velocity:
        result = model.sample_actions(**inputs, noise=noise, steps=steps)
    torch.testing.assert_close(result, target)
    torch.testing.assert_close(noise, original_noise, rtol=0, atol=0)
    assert velocity.call_count == steps


def test_euler_converges_for_state_dependent_field(siglip_dir):
    import math

    model = MiniMindVLA(small_config(siglip_dir)).eval()
    memory, valid = model.encode_context(**batch())
    noise = torch.ones(2, 4, 2)
    # dx/dt=x, x(0)=1의 정확한 해는 x(1)=e. 매 step 새 x를 평가해야 수렴한다.
    with patch.object(model, "velocity", side_effect=lambda memory, valid, x, time: x):
        coarse = model.integrate_actions(memory, valid, noise, steps=2)
        fine = model.integrate_actions(memory, valid, noise, steps=100)
    assert (fine - math.e).abs().max() < 0.02
    assert (fine - math.e).abs().max() < (coarse - math.e).abs().max()


def test_sampling_restores_mode_after_error(siglip_dir):
    model = MiniMindVLA(small_config(siglip_dir)).train()
    with pytest.raises(ValueError, match="positive integer"):
        model.sample_actions(**batch(), steps=True)
    assert model.training
    assert not model.vlm.vision_encoder.training


def test_all_modalities_change_prediction(siglip_dir):
    model = MiniMindVLA(
        small_config(
            siglip_dir,
        )
    ).eval()
    b = batch()
    noise = torch.randn(2, 4, 2)
    original = model.sample_actions(**b, noise=noise, steps=2)
    for key in ("pixel_values", "state", "input_ids"):
        changed = {k: v.clone() for k, v in b.items()}
        if key == "input_ids":
            changed[key][:, -3:-1] = 40
        else:
            changed[key] = torch.zeros_like(changed[key])
        assert not torch.allclose(
            original, model.sample_actions(**changed, noise=noise, steps=2)
        )


def test_text_padding_does_not_affect_output(siglip_dir):
    model = MiniMindVLA(
        small_config(
            siglip_dir,
        )
    ).eval()
    b = batch()
    b["attention_mask"][:, -2:] = False
    noise = torch.randn(2, 4, 2)
    expected = model.sample_actions(**b, noise=noise, steps=2)
    b["input_ids"][:, -2:] = 80
    torch.testing.assert_close(
        expected, model.sample_actions(**b, noise=noise, steps=2)
    )


def test_frozen_backbone_stays_eval_and_unchanged(siglip_dir):
    model = MiniMindVLA(small_config(siglip_dir, freeze_vlm=True)).train()
    before = {k: v.clone() for k, v in model.vlm.state_dict().items()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    model(**batch(), actions=torch.randn(2, 4, 2)).backward()
    opt.step()
    assert not model.vlm.training
    for k, v in model.vlm.state_dict().items():
        torch.testing.assert_close(before[k], v, rtol=0, atol=0)


def test_checkpoint_roundtrip_and_native_vlm_initialization(tmp_path, siglip_dir):
    model = MiniMindVLA(
        small_config(
            siglip_dir,
        )
    ).eval()
    b = batch()
    noise = torch.randn(2, 4, 2)
    expected = model.sample_actions(**b, noise=noise, steps=2)
    model.save_checkpoint(tmp_path / "vla.pt")
    loaded = MiniMindVLA.load_checkpoint(tmp_path / "vla.pt")
    torch.testing.assert_close(
        expected, loaded.sample_actions(**b, noise=noise, steps=2)
    )
    torch.save(model.vlm.state_dict(), tmp_path / "vlm.pth")
    other = MiniMindVLA(
        small_config(
            siglip_dir,
        )
    )
    other.load_vlm_weights(tmp_path / "vlm.pth")
    for k, v in model.vlm.state_dict().items():
        torch.testing.assert_close(v, other.vlm.state_dict()[k])
    torch.save({"bad": torch.zeros(1)}, tmp_path / "bad.pth")
    with pytest.raises((ValueError, RuntimeError)):
        other.load_vlm_weights(tmp_path / "bad.pth")


def test_reject_invalid_inputs(siglip_dir):
    model = MiniMindVLA(
        small_config(
            siglip_dir,
        )
    )
    b = batch()
    with pytest.raises(ValueError):
        model.sample_actions(**b, steps=0)
    with pytest.raises(ValueError):
        model(**b, actions=torch.randn(2, 3, 2))
    b["input_ids"][:, 0] = 1
    with pytest.raises(ValueError, match="image"):
        model.sample_actions(**b)


def test_native_siglip_matches_original_vlm_hidden_states(tmp_path, siglip_dir):
    from transformers import SiglipVisionConfig, SiglipVisionModel, SiglipImageProcessor
    from model.model_vlm import MiniMindVLM, VLMConfig
    from dataset.vla_dataset import VLAProcessor, collate_batch
    from PIL import Image

    directory = tmp_path / "siglip"
    vision = SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=32,
            patch_size=16,
        )
    )
    vision.save_pretrained(directory)
    SiglipImageProcessor(size={"height": 32, "width": 32}).save_pretrained(directory)
    config = small_config(siglip_dir, vision_model_path=str(directory))
    config.vlm_config.update(
        image_token_len=4, vocab_size=6400, max_position_embeddings=256
    )
    model = MiniMindVLA(config).eval()
    Image.new("RGB", (32, 32), "blue").save(tmp_path / "input.png")
    processor = VLAProcessor(config, model.vlm.processor)
    b = collate_batch([processor(tmp_path / "input.png", "move left", [0, 0])])
    original = MiniMindVLM(VLMConfig(**config.vlm_config), str(directory)).eval()
    original.load_state_dict(model.vlm.state_dict(), strict=True)
    with torch.no_grad():
        hidden = model.vlm(b["input_ids"], b["attention_mask"], b["pixel_values"])
        expected = original(
            input_ids=b["input_ids"],
            attention_mask=b["attention_mask"].float(),
            pixel_values={"pixel_values": b["pixel_values"]},
            logits_to_keep=1,
        ).hidden_states
    torch.testing.assert_close(hidden, expected)
    assert model.sample_actions(**b, steps=2).shape == (1, 4, 2)
    # Native trainers intentionally exclude the external frozen vision encoder.
    weights = {
        k: v
        for k, v in original.state_dict().items()
        if not k.startswith("vision_encoder.")
    }
    torch.save(weights, tmp_path / "native.pth")
    model.load_vlm_weights(tmp_path / "native.pth")
    assert not any(p.requires_grad for p in model.vlm.vision_encoder.parameters())
    model.train()
    assert not model.vlm.vision_encoder.training


def test_missing_siglip_fails(tmp_path, siglip_dir):
    with pytest.raises(ValueError, match="SigLIP"):
        MiniMindVLA(
            small_config(siglip_dir, vision_model_path=str(tmp_path / "missing"))
        )


def test_defaults_use_native_siglip_and_tokenizer():
    config = VLAConfig()
    assert config.vision_model_path != "tiny"
    assert config.vlm_config["image_hidden_size"] == 768
    assert config.vlm_config["image_token_len"] == 64
    assert config.vlm_config["vocab_size"] == 6400


def test_checkpoint_excludes_frozen_siglip(tmp_path, siglip_dir):
    model = MiniMindVLA(small_config(siglip_dir))
    model.save_checkpoint(tmp_path / "vla.pt")
    data = torch.load(tmp_path / "vla.pt", weights_only=True)
    assert not any(k.startswith("vlm.vision_encoder.") for k in data["model"])


def test_removed_tiny_encoder_is_rejected():
    with pytest.raises(ValueError, match="SigLIP"):
        VLAConfig(vision_model_path="tiny")


def test_incomplete_action_checkpoint_is_rejected(tmp_path, siglip_dir):
    model = MiniMindVLA(small_config(siglip_dir))
    path = tmp_path / "vla.pt"
    model.save_checkpoint(path)
    saved = torch.load(path, weights_only=True)
    del saved["model"]["action_out.weight"]
    torch.save(saved, path)
    with pytest.raises(ValueError, match="Incompatible VLA"):
        MiniMindVLA.load_checkpoint(path)


@pytest.mark.parametrize("depth, kept", [(1, 1), (2, 1), (5, 2), (8, 4)])
def test_only_first_half_layers_are_present_and_executed(siglip_dir, depth, kept):
    config = small_config(siglip_dir)
    config.vlm_config["num_hidden_layers"] = depth
    model = MiniMindVLA(config)
    assert len(model.vlm.model.layers) == kept
    assert model.vlm.config.num_hidden_layers == kept
    assert config.vlm_config["num_hidden_layers"] == depth
    calls = []
    handles = [
        layer.register_forward_hook(lambda *args: calls.append(1))
        for layer in model.vlm.model.layers
    ]
    try:
        model(**batch(), actions=torch.randn(2, 4, 2)).backward()
    finally:
        for handle in handles:
            handle.remove()
    assert len(calls) == kept
    for layer in model.vlm.model.layers:
        grad = layer.self_attn.q_proj.weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


def test_native_prefix_matches_truncated_original_and_roundtrips(tmp_path, siglip_dir):
    from model.model_vlm import MiniMindVLM, VLMConfig

    config = small_config(siglip_dir)
    config.vlm_config["num_hidden_layers"] = 4
    original = MiniMindVLM(VLMConfig(**config.vlm_config), str(siglip_dir)).eval()
    native_weights = {
        k: v
        for k, v in original.state_dict().items()
        if not k.startswith("vision_encoder.")
    }
    path = tmp_path / "native.pth"
    torch.save(native_weights, path)
    model = MiniMindVLA(config).eval()
    model.load_vlm_weights(path)
    assert len(model.vlm.model.layers) == 2
    for key, value in model.vlm.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)
    original.model.layers = original.model.layers[:2]
    b = batch()
    with torch.no_grad():
        expected = original(
            input_ids=b["input_ids"],
            attention_mask=b["attention_mask"].float(),
            pixel_values={"pixel_values": b["pixel_values"]},
        ).hidden_states
        actual = model.vlm(b["input_ids"], b["attention_mask"], b["pixel_values"])
    torch.testing.assert_close(actual, expected)
    noise = torch.randn(2, 4, 2)
    prediction = model.sample_actions(**b, noise=noise, steps=2)
    saved_path = tmp_path / "vla.pt"
    model.save_checkpoint(saved_path)
    saved = torch.load(saved_path, weights_only=True)
    assert saved["format_version"] == 2
    assert not any(k.startswith("vlm.model.layers.2.") for k in saved["model"])
    restored = MiniMindVLA.load_checkpoint(saved_path)
    assert len(restored.vlm.model.layers) == 2
    torch.testing.assert_close(
        prediction, restored.sample_actions(**b, noise=noise, steps=2)
    )
    # A missing retained key or an unrelated unexpected key must still fail.
    bad_weights = dict(native_weights)
    del bad_weights["model.layers.0.self_attn.q_proj.weight"]
    torch.save(bad_weights, path)
    with pytest.raises(ValueError, match="Incompatible VLM"):
        model.load_vlm_weights(path)
    bad_weights = dict(native_weights)
    bad_weights["model.layers.3.unknown"] = torch.zeros(1)
    torch.save(bad_weights, path)
    with pytest.raises(ValueError, match="Incompatible VLM"):
        model.load_vlm_weights(path)


def test_old_vla_format_requires_retraining(tmp_path, siglip_dir):
    model = MiniMindVLA(small_config(siglip_dir))
    path = tmp_path / "vla.pt"
    model.save_checkpoint(path)
    saved = torch.load(path, weights_only=True)
    saved.pop("format_version", None)
    torch.save(saved, path)
    with pytest.raises(ValueError, match="retrain"):
        MiniMindVLA.load_checkpoint(path)
