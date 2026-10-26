"""Full-question GRPO inputs; gold answers never enter model prompts."""
import json
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import Dataset
from trainer.gsm8k_reward import extract_gsm8k_answer

SUFFIX = '\nSolve step by step. End your answer with #### followed by the final number.'


class GSM8KDataset(Dataset):
    def __init__(self, source, tokenizer, max_length=768, revision=None):
        if Path(source).is_file():
            kind = 'parquet' if source.endswith('.parquet') else 'json'
            rows = load_dataset(kind, data_files=source, split='train')
        elif source == 'openai/gsm8k':
            rows = load_dataset(source, 'main', split='train', revision=revision)
        elif source == 'caspar/MiniMind-Mixed-SFT-Training-v1.0':
            rows = load_dataset(source, 'conversations', split='train', revision=revision or 'v1.0')
        else:
            raise ValueError('Use a local JSONL/Parquet file or a supported GSM8K repository')
        self.samples = []
        for index, row in enumerate(rows):
            if row.get('dataset', 'gsm8k') != 'gsm8k':
                continue
            if row.get('split', 'train') != 'train':
                raise ValueError('Only training rows may be used for GRPO')
            if 'messages_json' in row:
                messages = json.loads(row['messages_json'])
                if [m['role'] for m in messages] != ['user', 'assistant']:
                    raise ValueError(f'Expected a single question/answer pair at row {index}')
                question, answer = (m['content'] for m in messages)
            else:
                question, answer = row['question'], row['answer']
            if extract_gsm8k_answer(answer) is None:
                raise ValueError(f'Invalid GSM8K gold at row {index}')
            if not question.endswith(SUFFIX):
                question += SUFFIX
            prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': question}], tokenize=False,
                add_generation_prompt=True, open_thinking=False,
            ).replace('<think>\n\n</think>\n\n', '')
            if max_length and len(tokenizer(prompt, add_special_tokens=False).input_ids) > max_length:
                raise ValueError(f'Question at row {index} exceeds max_seq_len={max_length}; increase it instead of truncating the question')
            self.samples.append({'prompt': prompt, 'answer': answer})
        if not self.samples:
            raise ValueError('No GSM8K training rows found')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
