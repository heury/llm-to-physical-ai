"""JSONL: image path, instruction, normalized state and future action chunk."""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, SiglipImageProcessor


class VLAProcessor:
    def __init__(self, config, image_processor=None, tokenizer_path=None):
        self.config = config
        self.image_processor = image_processor or SiglipImageProcessor.from_pretrained(
            config.vision_model_path, local_files_only=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or str(Path(__file__).resolve().parents[1] / "model"),
            local_files_only=True,
        )
        self.n = config.vlm_config.get("image_token_len", 64)
        self.max_text = config.vlm_config.get("max_position_embeddings", 32768) - self.n
        if self.max_text < 2:
            raise ValueError("context must allow image and text tokens")

    def __call__(self, image_path, instruction, state):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be nonempty text")
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > self.max_text:
            raise ValueError(
                f"instruction too long: chat prompt has {len(ids)} tokens, "
                f"but only {self.max_text} fit after {self.n} image tokens; "
                "shorten the instruction or increase max_position_embeddings"
            )
        marker = self.config.vlm_config.get("image_ids", [12])[0]
        if marker in ids:
            raise ValueError("instruction contains reserved image marker")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            pixels = self.image_processor(images=image, return_tensors="pt")[
                "pixel_values"
            ][0]
        state = torch.tensor(state, dtype=torch.float32)
        if state.shape != (self.config.state_dim,) or not torch.isfinite(state).all():
            raise ValueError("invalid state dimensions or values")
        return dict(
            input_ids=torch.tensor([marker] * self.n + ids, dtype=torch.long),
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
        r = self.rows[index]
        c = self.config
        item = self.processor(
            self.path.parent / r["image"], r["instruction"], r["state"]
        )
        actions = torch.tensor(r["actions"], dtype=torch.float32)
        if (
            actions.ndim != 2
            or actions.shape[1] != c.action_dim
            or not 1 <= len(actions) <= c.chunk_size
        ):
            raise ValueError("actions must contain 1..chunk_size vectors of action_dim")
        if not torch.isfinite(actions).all():
            raise ValueError("actions must be finite")
        padded = torch.zeros(c.chunk_size, c.action_dim)
        padded[: len(actions)] = actions
        mask = torch.arange(c.chunk_size) < len(actions)
        return dict(**item, actions=padded, action_mask=mask)


def collate_batch(items):
    length = max(len(x["input_ids"]) for x in items)
    ids = torch.zeros(len(items), length, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, item in enumerate(items):
        n = len(item["input_ids"])
        ids[i, :n] = item["input_ids"]
        mask[i, :n] = True
    result = dict(input_ids=ids, attention_mask=mask)
    for k in ("pixel_values", "state", "actions", "action_mask"):
        if k in items[0]:
            result[k] = torch.stack([x[k] for x in items])
    return result
