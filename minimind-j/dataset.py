"""One shared context + isolated question branches, packed into one sequence per record."""
import json
import string
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ByteTokenizer:
    """Offline UTF-8 byte tokenizer; Korean letters use multiple tokens."""
    pad_token_id, bos_token_id, eos_token_id, sep_token_id = 0, 1, 2, 3
    vocab_size = 260

    def encode(self, text, add_special_tokens=False):
        return [byte + 4 for byte in text.encode("utf-8")]

    def __len__(self):
        return self.vocab_size


def normalize_row(row):
    """Accept v2 typed questions, or the old single-choice JSONL convenience format."""
    if not isinstance(row, dict):
        raise ValueError("Each row must be an object")
    if not isinstance(row.get("input"), str) or not row["input"].strip():
        raise ValueError("input must be a nonempty string")
    if "questions" in row:
        return row
    choices = row.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Provide questions or a nonempty choices list")
    question = {"id": "answer", "type": "choice", "question": "Choose the best option.", "choices": choices}
    if "label" in row:
        label = row["label"]
        if type(label) is not int or not 0 <= label < len(choices):
            raise ValueError("label must be a zero-based integer choice index")
        question["answer"] = choices[label]
    return {"input": row["input"], "questions": [question]}


def answer_values(question):
    """Code A/B/... maps to these typed JSON values; score is ordinal, not regression."""
    if question["type"] == "choice":
        return question["choices"]
    if question["type"] == "score":
        return list(range(question["min"], question["max"] + 1))
    return [False, True]  # noul: the boolean type name in the reference implementation


def validate_row(row, require_label=True):
    row = normalize_row(row)
    questions = row.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a nonempty list")
    seen = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Each question must be an object")
        qid = question.get("id")
        if not isinstance(qid, str) or not qid.strip() or qid in seen:
            raise ValueError("Question id must be nonempty and unique within a record")
        seen.add(qid)
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError(f"{qid}: question must be a nonempty string")
        kind = question.get("type")
        if kind not in ("choice", "score", "noul"):
            raise ValueError(f"{qid}: type must be choice, score or noul")
        if kind == "choice":
            choices = question.get("choices")
            if (not isinstance(choices, list) or not choices
                    or any(not isinstance(c, str) or not c.strip() for c in choices)
                    or len(set(choices)) != len(choices)):
                raise ValueError(f"{qid}: choices must be nonempty, unique strings")
            if len(choices) > 26:
                raise ValueError(f"{qid}: at most 26 choices (A-Z)")
        elif kind == "score":
            low, high = question.get("min"), question.get("max")
            if type(low) is not int or type(high) is not int or not 1 <= high - low + 1 <= 26:
                raise ValueError(f"{qid}: score needs integer min <= max, at most 26 levels")
        if require_label and "answer" not in question:
            raise ValueError(f"{qid}: training requires answer")
        if "answer" in question:
            answer = question["answer"]
            valid_type = {"choice": str, "score": int, "noul": bool}[kind]
            if type(answer) is not valid_type or answer not in answer_values(question):
                raise ValueError(f"{qid}: answer has wrong type or is outside allowed values")
    return row


class DecisionDataset(Dataset):
    def __init__(self, path):
        self.rows = []
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    self.rows.append(validate_row(json.loads(line)))
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
        if not self.rows:
            raise ValueError("Dataset is empty")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class DecisionCollator:
    def __init__(self, tokenizer, max_length=512):
        if max_length < 4:
            raise ValueError("max_length must be at least 4")
        self.tokenizer, self.max_length = tokenizer, max_length
        for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
            if getattr(tokenizer, name, None) is None:
                raise ValueError(f"Tokenizer needs {name}")

    def __call__(self, rows):
        if not rows:
            raise ValueError("Empty batch")
        rows = [validate_row(row, require_label=False) for row in rows]
        tok = self.tokenizer
        sep = getattr(tok, "sep_token_id", None)
        sep = tok.eos_token_id if sep is None else sep
        # Code tokens must be single, distinct vocabulary entries for this tokenizer.
        count = max(len(answer_values(q)) for row in rows for q in row["questions"])
        codes = string.ascii_uppercase[:count]
        code_ids = [tok.encode(code, add_special_tokens=False) for code in codes]
        if (any(len(ids) != 1 for ids in code_ids)
                or len({ids[0] for ids in code_ids}) != count
                or any(ids[0] == getattr(tok, "unk_token_id", None) for ids in code_ids)):
            raise ValueError("Tokenizer must encode each answer code A-Z as a distinct single token")
        code_ids = [ids[0] for ids in code_ids]
        sequences, positions, layouts, queries, candidates, labels = [], [], [], [], [], []
        for row_index, row in enumerate(rows):
            ids = [tok.bos_token_id] + tok.encode(row["input"], add_special_tokens=False) + [sep]
            prefix_length = len(ids)
            pos = list(range(prefix_length))
            spans = []
            for question in row["questions"]:
                values = answer_values(question)
                options = "\n".join(f"{code}: {json.dumps(value, ensure_ascii=False)}" for code, value in zip(codes, values))
                text = f"\nType: {question['type']}\nQuestion: {question['question']}\n{options}\nAnswer:"
                branch = tok.encode(text, add_special_tokens=False)
                if not branch:
                    raise ValueError("Tokenizer produced an empty question branch")
                begin = len(ids)
                ids.extend(branch)
                # Local positions keep answers invariant to unrelated question order/length.
                pos.extend(range(prefix_length, prefix_length + len(branch)))
                spans.append((begin, len(ids)))
                queries.append((row_index, len(ids) - 1))
                candidates.append(code_ids[:len(values)])
                labels.append(values.index(question["answer"]) if "answer" in question else -100)
            if len(ids) > self.max_length:
                raise ValueError(f"Packed record length {len(ids)} exceeds max_length={self.max_length}; shorten text or increase max_length")
            sequences.append(ids)
            positions.append(pos)
            layouts.append((prefix_length, spans))
        length = max(map(len, sequences))
        input_ids = torch.full((len(rows), length), tok.pad_token_id, dtype=torch.long)
        position_ids = torch.zeros_like(input_ids)
        # True = allowed. Padding queries see themselves to prevent all-masked softmax NaNs.
        attention_mask = torch.eye(length, dtype=torch.bool).expand(len(rows), 1, length, length).clone()
        for i, (ids, pos, (prefix, spans)) in enumerate(zip(sequences, positions, layouts)):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            position_ids[i, :len(pos)] = torch.tensor(pos)
            attention_mask[i, 0, :prefix, :prefix] = torch.ones(prefix, prefix, dtype=torch.bool).tril()
            for begin, end in spans:
                attention_mask[i, 0, begin:end, :prefix] = True
                attention_mask[i, 0, begin:end, begin:end] = torch.ones(end-begin, end-begin, dtype=torch.bool).tril()
        option_mask = torch.zeros(len(queries), count, dtype=torch.bool)
        answer_token_ids = torch.zeros(len(queries), count, dtype=torch.long)
        for i, candidate in enumerate(candidates):
            option_mask[i, :len(candidate)] = True
            answer_token_ids[i, :len(candidate)] = torch.tensor(candidate)
        return dict(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                    question_positions=torch.tensor(queries), answer_token_ids=answer_token_ids,
                    option_mask=option_mask, labels=torch.tensor(labels))
