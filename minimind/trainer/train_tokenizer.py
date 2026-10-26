# 참고: MiniMind에 토크나이저가 포함되어 있으므로 재학습은 권장하지 않습니다. 이 스크립트는 학습과 참고용입니다. 서로 다른 어휘 목록으로 학습한 모델은 출력이 달라져 커뮤니티에서 모델을 재사용하기 어려워집니다.
# Note: It is not recommended to re-train the tokenizer. MiniMind already includes one. This script is for learning and reference only. Training models with different tokenizers will lead to inconsistent outputs and reduce model reusability in the community.
import os
import json
import argparse
import time
from pathlib import Path
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

DATA_PATH = '../dataset/sft_t2t_mini.jsonl'
TOKENIZER_DIR = '../model_learn_tokenizer/'
VOCAB_SIZE = 6400
SPECIAL_TOKENS_NUM = 36
MAX_LINES = 0  # 0이면 전체를 읽고, 양수이면 처음 N개만 사용(빠른 시험 실행)

def get_hf_texts(dataset_name, split='train', revision=None, max_lines=MAX_LINES, stats=None):
    """Stream a Hugging Face text dataset without materializing the full corpus."""
    from datasets import load_dataset
    records = load_dataset(dataset_name, split=split, revision=revision, streaming=True)
    stats = {} if stats is None else stats
    stats.update({'dataset_rows': 0, 'skipped_empty_rows': 0})
    used = 0
    for record in records:
        stats['dataset_rows'] += 1
        text = record.get('text')
        if not isinstance(text, str) or not text.strip():
            stats['skipped_empty_rows'] += 1
            continue
        yield text
        used += 1
        if max_lines and used >= max_lines:
            break
    if used == 0:
        raise ValueError(f'{dataset_name}: no non-empty strings in the text column')

def get_texts(data_path, max_lines=MAX_LINES):
    """BPE 학습용 텍스트를 한 줄씩 반환하며 두 가지 주요 데이터 형식을 지원합니다.

    pretrain_t2t(_mini).jsonl 형식: {"text": ...}
    sft_t2t(_mini).jsonl 형식: {"conversations": [{"role":..., "content":...}, ...]}
    기존에는 conversations만 파싱하여 DATA_PATH에 사전 학습 데이터를 지정하면
    텍스트를 하나도 읽지 못했습니다. 이 경우 BPE는 initial_alphabet만으로 학습되어
    병합 규칙이 없는 어휘 목록을 생성하면서도 오류를 보고하지 않았습니다.
    """
    used = 0
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if max_lines and used >= max_lines:
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'text' in data:
                text = str(data['text'])
            else:
                contents = [item.get('content') for item in data.get('conversations', []) if item.get('content')]
                text = "\n".join(contents) if contents else ''
            if text.strip():
                used += 1
                yield text
    if used == 0:
        raise ValueError(
            f'{data_path}에 사용 가능한 텍스트가 없음: 각 줄은 다음 형식이어야 함 {{"text": ...}} 또는 '
            f'{{"conversations": [{{"role":..., "content":...}}, ...]}}.'
            f'학습을 계속하면 빈 어휘 목록만 생성됩니다.'
        )

def train_tokenizer(data_path, tokenizer_dir, vocab_size, special_tokens_num=SPECIAL_TOKENS_NUM, texts=None):
    tokenizer_dir = os.fspath(tokenizer_dir)
    if special_tokens_num < 27 or vocab_size < special_tokens_num + 256:
        raise ValueError('Vocabulary must fit the reserved tokens and all 256 byte symbols')
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    special_tokens_list = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", 
        "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", 
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", 
        "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>"
    ]
    
    additional_tokens_list = [
        "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
        "<think>", "</think>"
    ]
    num_buffer = special_tokens_num - len(special_tokens_list + additional_tokens_list)
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)] # 일정 수의 토큰 위치를 미리 확보
    all_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=all_special_tokens
    )
    texts = get_texts(data_path) if texts is None else texts
    stats = {'documents': 0, 'characters': 0, 'requested_vocab_size': vocab_size}
    started = time.monotonic()

    def counted_texts():
        for text in texts:
            stats['documents'] += 1
            stats['characters'] += len(text)
            if stats['documents'] % 10000 == 0:
                print(f"Read {stats['documents']:,} documents / {stats['characters']:,} characters "
                      f"in {time.monotonic() - started:.0f}s", flush=True)
            yield text
        if stats['documents'] == 0:
            raise ValueError('No training texts were provided')

    tokenizer.train_from_iterator(counted_texts(), trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(special_tokens_list)

    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    for token_info in tokenizer_data.get('added_tokens', []):
        if token_info['content'] not in special_tokens_list:
            token_info['special'] = False
    with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    added_tokens_decoder = {}
    for i, token in enumerate(all_special_tokens):
        idx = tokenizer.token_to_id(token)
        added_tokens_decoder[str(idx)] = {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True if token in special_tokens_list else False
        }

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": [t for t in special_tokens_list if t not in ["<|endoftext|>"]],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 131072,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "unk_token": "<|endoftext|>",
        "image_token": "<|image_pad|>",
        "audio_token": "<|audio_pad|>",
        "video_token": "<|video_pad|>",
        "vision_bos_token": "<|vision_start|>",
        "vision_eos_token": "<|vision_end|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if true %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if open_thinking is defined and open_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}",
        "tokenizer_class": "PreTrainedTokenizerFast"
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    stats['vocab_size'] = tokenizer.get_vocab_size()
    stats['elapsed_seconds'] = round(time.monotonic() - started, 2)
    with open(os.path.join(tokenizer_dir, 'training_info.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("Tokenizer training completed.")

def eval_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    messages = [
        {"role": "system", "content": "너는 항상 정확한 답변을 제공하는 훌륭한 챗봇이야!"},
        {"role": "user", "content": '너는 어디에서 왔니?'},
        {"role": "assistant", "content": '나는 달에서 왔어'},
        {"role": "user", "content": '너는 도대체 어디에서 왔니?'},
        {"role": "assistant", "content": '나는 지구에서 왔어'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print('-'*100)
    print(new_prompt)
    print('-'*100)
    print('토크나이저 어휘 수: ', len(tokenizer))
    model_inputs = tokenizer(new_prompt)
    print('인코딩 길이: ', len(model_inputs['input_ids']))
    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
    print('디코더 일치 여부: ', response == new_prompt, "\n")
    print('-'*100)
    print('압축률 테스트(문자 수/토큰 수): ')
    test_texts = [
        # 영어 샘플(약 200단어/자)
        "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast amounts of text data to understand and generate human-like language. These models use deep learning techniques, specifically transformers, to process and predict the next word in a sequence. LLMs like GPT-4, Llama, and Claude have demonstrated remarkable capabilities in coding, translation, and creative writing. However, they also face challenges such as hallucinations, where the model generates factually incorrect information, and the need for significant computational resources.",
        "The development of sustainable energy is crucial for the future of our planet. As climate change continues to impact global weather patterns, transitioning from fossil fuels to renewable sources like solar, wind, and hydroelectric power has become an urgent priority. Innovations in battery storage technology and smart grid management are essential to ensure a reliable energy supply. International cooperation and policy frameworks are also necessary to drive the global shift towards a greener economy and reduce carbon emissions.",
    ]
    
    total_compression = 0
    for i, text in enumerate(test_texts):
        encoded = tokenizer.encode(text)
        token_count = len(encoded)
        char_count = len(text)
        compression_ratio = char_count / token_count
        total_compression += compression_ratio
        print(f"샘플 {i+1} | 문자 수: {char_count:4} | Tokens: {token_count:3} | 압축률: {compression_ratio:.2f}")
    
    print(f"평균 압축률: {total_compression / len(test_texts):.2f}")
    print('-'*100)
    print('스트리밍 디코딩(바이트 버퍼) 테스트: ')
    input_ids = model_inputs['input_ids']
    token_cache = []
    for tid in input_ids:
        token_cache.append(tid)
        current_decode = tokenizer.decode(token_cache)
        if current_decode and '\ufffd' not in current_decode:
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            raw_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in (token_cache if isinstance(token_cache, list) else [token_cache])]
            print(f'Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}')
            token_cache = []

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='Train MiniMind byte-level BPE from JSONL or a streamed HF dataset')
    parser.add_argument('--data-path', default=str(root / 'dataset/sft_t2t_mini.jsonl'))
    parser.add_argument('--dataset', help='Hugging Face dataset ID; uses the text column')
    parser.add_argument('--revision', help='Dataset commit hash for reproducible training')
    parser.add_argument('--split', default='train')
    parser.add_argument('--tokenizer-dir', default=str(root / 'model_learn_tokenizer'))
    parser.add_argument('--vocab-size', type=int, default=VOCAB_SIZE)
    parser.add_argument('--max-lines', type=int, default=MAX_LINES, help='0 = all usable documents')
    parser.add_argument('--skip-eval', action='store_true')
    args = parser.parse_args()
    if args.max_lines < 0:
        parser.error('--max-lines must be non-negative')
    input_stats = {}
    texts = (get_hf_texts(args.dataset, args.split, args.revision, args.max_lines, stats=input_stats)
             if args.dataset else get_texts(args.data_path, args.max_lines))
    train_tokenizer(args.data_path, args.tokenizer_dir, args.vocab_size, texts=texts)
    info_path = Path(args.tokenizer_dir) / 'training_info.json'
    info = json.loads(info_path.read_text(encoding='utf-8'))
    info.update(input_stats)
    info.update({'dataset': args.dataset, 'revision': args.revision, 'split': args.split,
                 'max_lines': args.max_lines, 'data_path': None if args.dataset else args.data_path})
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if not args.skip_eval:
        eval_tokenizer(args.tokenizer_dir)
