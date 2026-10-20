import json
from PIL import Image
import pytest
from dataset.vla_dataset import VLADataset, VLAProcessor, collate_batch
from model.model_vla import VLAConfig
from transformers import SiglipImageProcessor


def test_instruction_length_boundary_preserves_all_tokens(tmp_path, siglip_dir):
    config = VLAConfig(
        vision_model_path=str(siglip_dir),
        vlm_config=dict(image_token_len=16, max_position_embeddings=128),
    )
    processor = VLAProcessor(config)
    instruction = "move to the red target"
    prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    expected = processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    config.vlm_config["max_position_embeddings"] = 16 + len(expected)
    processor = VLAProcessor(config)
    image = tmp_path / "image.png"
    Image.new("RGB", (32, 32)).save(image)
    assert processor(image, instruction, [0, 0])["input_ids"].tolist() == (
        [12] * 16 + expected
    )
    config.vlm_config["max_position_embeddings"] -= 1
    with pytest.raises(ValueError, match="too long"):
        VLAProcessor(config)(image, instruction, [0, 0])


@pytest.mark.parametrize("direction", ["left", "right"])
def test_long_instruction_is_rejected_not_silently_truncated(
    tmp_path, siglip_dir, direction
):
    config = VLAConfig(vision_model_path=str(siglip_dir))
    image = tmp_path / "image.png"
    Image.new("RGB", (32, 32)).save(image)
    instruction = "look at the scene carefully. " * 100 + f"move {direction}"
    with pytest.raises(ValueError, match="too long"):
        VLAProcessor(config)(image, instruction, [0, 0])


def test_real_image_variable_length_actions_and_padding(tmp_path, siglip_dir):
    Image.new("RGB", (32, 32), "red").save(tmp_path / "red.png")
    records = [
        dict(
            image="red.png",
            instruction="go right",
            state=[0, 0],
            actions=[[0.1, 0], [0.2, 0]],
        ),
        dict(
            image="red.png",
            instruction="왼쪽으로 이동",
            state=[0, 0],
            actions=[[0, 0.1]],
        ),
    ]
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    ds = VLADataset(
        path,
        VLAConfig(
            chunk_size=4,
            vision_model_path=str(siglip_dir),
            vlm_config=dict(image_token_len=16),
        ),
        SiglipImageProcessor.from_pretrained(siglip_dir),
    )
    b = collate_batch([ds[0], ds[1]])
    assert b["pixel_values"].shape == (2, 3, 64, 64)
    assert b["actions"].shape == (2, 4, 2)
    assert b["action_mask"].sum() == 3
    assert b["attention_mask"].sum(1)[0] < b["attention_mask"].sum(1)[1]
    assert b["input_ids"].max() < 6400
    assert (b["input_ids"][:, :16] == 12).all()
    assert b["pixel_values"][0, 0].mean() > b["pixel_values"][0, 1].mean()


@pytest.mark.parametrize(
    "bad",
    [
        {"actions": []},
        {"actions": [[1, 2, 3]]},
        {"state": [0]},
        {"state": [float("nan"), 0]},
        {"instruction": ""},
    ],
)
def test_bad_records_fail_clearly(tmp_path, bad, siglip_dir):
    Image.new("RGB", (32, 32)).save(tmp_path / "x.png")
    r = dict(image="x.png", instruction="move", state=[0, 0], actions=[[0, 0]])
    r.update(bad)
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        VLADataset(
            p,
            VLAConfig(vision_model_path=str(siglip_dir)),
            SiglipImageProcessor.from_pretrained(siglip_dir),
        )[0]
