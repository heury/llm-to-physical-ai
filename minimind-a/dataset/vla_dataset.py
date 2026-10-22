"""JSONL 한 줄을 이미지·지시문·상태·정답 행동으로 변환하고 배치로 묶는다."""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer, SiglipImageProcessor


class VLAProcessor:
    """학습·추론에서 동일한 이미지 처리와 지시문 토큰화를 사용한다."""

    def __init__(self, config, image_processor=None, tokenizer_path=None):
        self.config = config
        self.image_processor = image_processor or SiglipImageProcessor.from_pretrained(
            config.vision_model_path, local_files_only=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or str(Path(__file__).resolve().parents[1] / "model"),
            local_files_only=True,
        )
        self.image_token_count = config.vlm_config.get("image_token_len", 64)
        self.max_text_tokens = (
            config.vlm_config.get("max_position_embeddings", 32768)
            - self.image_token_count
        )
        if self.max_text_tokens < 2:
            raise ValueError("context must allow image and text tokens")

    def encode_instruction(self, instruction):
        """이미지 자리표시자 + chat template를 적용한 지시문 → 토큰 [L]."""
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be nonempty text")
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > self.max_text_tokens:
            raise ValueError(
                f"instruction too long: chat prompt has {len(ids)} tokens, "
                f"but only {self.max_text_tokens} fit after {self.image_token_count} image tokens; "
                "shorten the instruction or increase max_position_embeddings"
            )
        marker = self.config.vlm_config.get("image_ids", [12])[0]
        if marker in ids:
            raise ValueError("instruction contains reserved image marker")
        return torch.tensor([marker] * self.image_token_count + ids, dtype=torch.long)

    def __call__(self, image_path, instruction, state):
        input_ids = self.encode_instruction(instruction)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            pixels = self.image_processor(images=image, return_tensors="pt")[
                "pixel_values"
            ][0]
        state = torch.tensor(state, dtype=torch.float32)
        if state.shape != (self.config.state_dim,) or not torch.isfinite(state).all():
            raise ValueError("invalid state dimensions or values")
        return dict(
            input_ids=input_ids,
            pixel_values=pixels,
            state=state,
        )


class VLADataset(Dataset):
    def __init__(self, path, config, image_processor=None):
        self.path = Path(path)
        self.rows = [
            json.loads(line)
            for line in self.path.read_text().splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError("empty dataset")
        self.config = config
        self.processor = VLAProcessor(config, image_processor)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        config = self.config
        item = self.processor(
            self.path.parent / row["image"], row["instruction"], row["state"]
        )
        actions = torch.tensor(row["actions"], dtype=torch.float32)
        if (
            actions.ndim != 2
            or actions.shape[1] != config.action_dim
            or not 1 <= len(actions) <= config.chunk_size
        ):
            raise ValueError("actions must contain 1..chunk_size vectors of action_dim")
        if not torch.isfinite(actions).all():
            raise ValueError("actions must be finite")
        padded = torch.zeros(config.chunk_size, config.action_dim)
        padded[: len(actions)] = actions
        mask = torch.arange(config.chunk_size) < len(actions)
        return dict(**item, actions=padded, action_mask=mask)


def pad_tokens(sequences):
    """오른쪽을 패딩하고 실제 토큰 위치가 True인 mask를 반환한다."""
    ids = pad_sequence(sequences, batch_first=True, padding_value=0)
    lengths = ids.new_tensor([len(sequence) for sequence in sequences])
    mask = torch.arange(ids.size(1), device=ids.device)[None] < lengths[:, None]
    return ids, mask


def collate_batch(items):
    """길이가 다른 지시문은 패딩하고, 고정 크기 이미지·상태·행동은 쌓는다."""
    ids, mask = pad_tokens([item["input_ids"] for item in items])
    result = dict(input_ids=ids, attention_mask=mask)
    for k in ("pixel_values", "state", "actions", "action_mask"):
        if k in items[0]:
            result[k] = torch.stack([x[k] for x in items])
    return result
