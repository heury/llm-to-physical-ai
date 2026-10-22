import json
import sys
from dataclasses import asdict

import pytest
import torch
from PIL import Image

import train_vla
import eval_vla
from model.model_vla import MiniMindVLA
from model.model_vlm import MiniMindVLM, VLMConfig
from test_vla import small_config


def test_frozen_training_requires_checkpoint_before_loading_vision(
    tmp_path, monkeypatch, capsys
):
    config = small_config(tmp_path / "missing-vision", freeze_vlm=True)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(config)))
    monkeypatch.setattr(sys, "argv", ["train_vla.py", "--config", str(path)])
    with pytest.raises(SystemExit) as error:
        train_vla.main()
    assert error.value.code == 2
    assert "--vlm-checkpoint" in capsys.readouterr().err


@pytest.mark.parametrize("freeze_vlm", [False, True])
def test_training_accepts_random_trainable_or_pretrained_frozen_vlm(
    tmp_path, siglip_dir, monkeypatch, freeze_vlm
):
    config = small_config(siglip_dir, freeze_vlm=freeze_vlm)
    config.vlm_config["num_hidden_layers"] = 4
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)))
    Image.new("RGB", (64, 64), "red").save(tmp_path / "image.png")
    data = tmp_path / "data.jsonl"
    data.write_text(
        json.dumps(
            dict(
                image="image.png",
                instruction="move left",
                state=[0, 0],
                actions=[[0.1, 0.2]],
            )
        )
        + "\n"
    )
    output = tmp_path / "output"
    args = [
        "train_vla.py",
        "--config",
        str(config_path),
        "--data",
        str(data),
        "--val-data",
        str(data),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--output",
        str(output),
    ]
    if freeze_vlm:
        original = MiniMindVLM(VLMConfig(**config.vlm_config), str(siglip_dir))
        checkpoint = tmp_path / "native.pth"
        torch.save(original.state_dict(), checkpoint)
        args += ["--vlm-checkpoint", str(checkpoint)]
    monkeypatch.setattr(sys, "argv", args)
    train_vla.main()
    restored = MiniMindVLA.load_checkpoint(output / "vla.pt")
    assert len(restored.vlm.model.layers) == 2
    history = json.loads((output / "history.json").read_text())
    assert torch.isfinite(torch.tensor(history[-1]["train_loss"]))
    assert all(torch.isfinite(torch.tensor(row["val_loss"])) for row in history)
    if freeze_vlm:
        for key, value in restored.vlm.state_dict().items():
            torch.testing.assert_close(
                value, original.state_dict()[key], rtol=0, atol=0
            )
    # 저장된 모델을 추론 CLI로 읽어 실제 JSONL → 행동 → JSON 출력까지 검사한다.
    prediction = output / "prediction.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_vla.py",
            "--checkpoint",
            str(output / "vla.pt"),
            "--data",
            str(data),
            "--output",
            str(prediction),
            "--steps",
            "2",
        ],
    )
    eval_vla.main()
    result = json.loads(prediction.read_text())
    assert torch.tensor(result["actions"]).shape == (
        config.chunk_size,
        config.action_dim,
    )
    assert torch.isfinite(torch.tensor(result["actions"])).all()
    assert torch.isfinite(torch.tensor(result["mse"]))
