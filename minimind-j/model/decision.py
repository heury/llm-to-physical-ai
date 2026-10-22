"""One packed MiniMind forward, existing LM head, parallel typed answers."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from dataset import ByteTokenizer, answer_values, validate_row

from .model_minimind import MiniMindConfig, MiniMindModel


class DecisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = MiniMindConfig(**config.to_dict())
        config = self.config
        if config.use_moe:
            raise ValueError("This educational model supports dense MiniMind only")
        if (config.hidden_size <= 0 or config.num_hidden_layers < 1
                or config.num_attention_heads < 1 or config.num_key_value_heads < 1
                or config.num_attention_heads % config.num_key_value_heads
                or config.head_dim < 2 or config.head_dim % 2
                or config.vocab_size < 1 or config.intermediate_size < 1
                or config.max_position_embeddings < 1):
            raise ValueError("Invalid MiniMind dimensions / GQA heads / RoPE head_dim")
        self.backbone = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embed_tokens.weight

    def forward(self, input_ids, attention_mask, position_ids, question_positions,
                answer_token_ids, option_mask, labels=None):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               position_ids=position_ids)
        # Every question has one answer position; no autoregressive answer generation.
        pooled = hidden[question_positions[:, 0], question_positions[:, 1]]
        # Exactly the A-Z rows of the existing LM projection, avoiding a full vocab tensor.
        weights = self.lm_head.weight[answer_token_ids]
        logits = (pooled.unsqueeze(1) * weights).sum(-1).float()
        logits = logits.masked_fill(~option_mask, float("-inf"))
        loss = None
        if labels is not None and (labels != -100).any():
            loss = F.cross_entropy(logits, labels, ignore_index=-100)
        return {"logits": logits, "probabilities": logits.softmax(-1), "loss": loss}

    @staticmethod
    def decode(rows, probabilities):
        # The tensor uses record order, then question order, as the collator does.
        rows = [validate_row(row, require_label=False) for row in rows]
        if len(probabilities) != sum(len(row["questions"]) for row in rows):
            raise ValueError("Probability rows do not match question count")
        result, cursor = [], 0
        for row in rows:
            answers, details = {}, {}
            for question in row["questions"]:
                values = answer_values(question)
                p = probabilities[cursor, :len(values)].detach().cpu()
                if len(p) != len(values):
                    raise ValueError("Too few probability columns")
                index = int(p.argmax())
                qid = question["id"]
                answers[qid] = values[index]
                details[qid] = {"type": question["type"], "values": values, "probabilities": p.tolist()}
                cursor += 1
            result.append({"answers": answers, "details": details})
        return result

    def freeze_backbone(self):
        # A tied LM projection shares the input embeddings. Untie before freezing so
        # head-only training cannot alter the frozen encoder's embedding matrix.
        if self.lm_head.weight is self.backbone.embed_tokens.weight:
            self.lm_head.weight = nn.Parameter(self.lm_head.weight.detach().clone())
            self.config.tie_word_embeddings = False
        self.backbone.requires_grad_(False)

    def load_lm_state_dict(self, state):
        """Native MiniMind LM weights, including its LM head; require exact compatibility."""
        if self.config.tie_word_embeddings and "lm_head.weight" in state and "model.embed_tokens.weight" in state:
            if not torch.equal(state["lm_head.weight"], state["model.embed_tokens.weight"]):
                raise RuntimeError("Tied config but LM input/output embeddings differ")
        renamed = {"backbone." + k.removeprefix("model.") if k.startswith("model.") else k: v
                   for k, v in state.items()}
        self.load_state_dict(renamed, strict=True)


def save_checkpoint(folder, model, tokenizer, max_length):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not isinstance(tokenizer, ByteTokenizer):
        tokenizer.save_pretrained(folder / "tokenizer")
    metadata = {"format_version": 2, "layout": "parallel_questions_lm_head", "config": model.config.to_dict(),
                "tokenizer": "byte" if isinstance(tokenizer, ByteTokenizer) else "hf", "max_length": max_length}
    (folder / "decision_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), folder / "decision.pth")


def load_checkpoint(folder, device="cpu"):
    folder = Path(folder)
    metadata = json.loads((folder / "decision_config.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != 2:
        raise ValueError("Unsupported checkpoint format: v1 used a different head; retrain with v2")
    if metadata.get("layout") != "parallel_questions_lm_head" or metadata.get("tokenizer") not in ("byte", "hf"):
        raise ValueError("Unsupported checkpoint layout/tokenizer")
    model = DecisionModel(MiniMindConfig(**metadata["config"]))
    model.load_state_dict(torch.load(folder / "decision.pth", map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    tokenizer = ByteTokenizer() if metadata["tokenizer"] == "byte" else AutoTokenizer.from_pretrained(folder / "tokenizer", local_files_only=True)
    return model, tokenizer, metadata["max_length"]
