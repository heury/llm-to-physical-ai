import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import re
import gc
import json
import math
import random
import argparse
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import AgentRLDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel, safe_math_eval
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps

warnings.filterwarnings('ignore')

# ================================ 도구 및 Reward = Start ================================

def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

# ======== 도구 정의 ========
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "Evaluate a mathematical expression", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "Convert units", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "Get current time", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "Get exchange rate", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "Translate text", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
]

# ======== 모의 데이터 ========
WEATHER_DATA = {"Beijing": ("28°C", "Clear"), "Shanghai": ("15°C", "Cloudy"), "Guangzhou": ("32°C", "Hot and humid"), "Shenzhen": ("30°C", "Clear"), "Hangzhou": ("22°C", "Overcast"), "Chengdu": ("18°C", "Light rain"), "Wuhan": ("25°C", "Cloudy"), "Nanjing": ("20°C", "Clear"), "Xi'an": ("16°C", "Strong winds"), "Chongqing": ("26°C", "Overcast"), "Tokyo": ("12°C", "Clear"), "New York": ("8°C", "Cloudy"), "London": ("5°C", "Light rain"), "Paris": ("10°C", "Overcast"), "Sydney": ("25°C", "Clear")}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00", "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00", "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00"}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72}
# TRANSLATE_DATA = {("Hello World", "english"): "Hello World", ("Good morning", "korean"): "Good morning", ("The weather is really nice today", "english"): "The weather is nice today", ("I love programming", "korean"): "I love programming", ("Machine learning is interesting", "english"): "Machine learning is interesting", ("Happy birthday", "korean"): "Happy birthday"}
UNIT_DATA = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048, "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556}

def convert_unit(arguments):
    value = float(arguments.get('value', 0))
    pair = f"{arguments.get('from_unit', '').lower()}_{arguments.get('to_unit', '').lower()}"
    if pair == 'celsius_fahrenheit':
        result = value * 1.8 + 32
    elif pair == 'fahrenheit_celsius':
        result = (value - 32) / 1.8
    else:
        result = value * UNIT_DATA.get(pair, 1)
    return {'result': round(result, 4)}

# ======== 모의 실행 ========
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(safe_math_eval(args.get("expression", "0")))},
    "unit_converter": convert_unit,
    "get_current_weather": lambda args: (lambda w: {"city": args.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1]})(WEATHER_DATA.get(args.get("location"), ("22°C", "Clear"))),
    "get_current_time": lambda args: {"datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency"), "to": args.get("to_currency"), "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")), 1.0)},
    # "translate_text": lambda args: {"translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", ""))},
}

# ======== 매개변수 검증 ========
CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time": lambda a: True,
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}

# ======== 도구 호출 파싱 및 실행 ========
def parse_tool_calls(text):
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try:
            call = json.loads(m.strip())
            if isinstance(call, dict) and isinstance(call.get("name"), str) and isinstance(call.get("arguments", {}), (dict, str)):
                calls.append(call)
        except (ValueError, TypeError):
            pass
    return calls

def execute_tool(name, args):
    fn = MOCK_RESULTS.get(name)
    if not fn: return None
    try:
        return fn(args)
    except Exception:
        return None

# ======== 여러 턴 Rollout ========
def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    all_outputs = []
    prompt_ids = None
    response_ids = []
    response_mask = []
    response_old_logps = []
    final_context = ""
    unfinished = False
    open_thinking = random.random() < thinking_ratio
    for turn in range(max_turns):
        context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
        if prompt_ids is None:
            prompt_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([prompt_ids + response_ids], device=device)
        rollout_result = rollout_engine.rollout(
            prompt_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )
        valid_len = int(rollout_result.completion_mask[0].sum().item())
        new_ids = rollout_result.completion_ids[0, :valid_len].tolist()
        new_logps = rollout_result.per_token_logps[0, :valid_len].tolist()
        if len(new_ids) != len(new_logps):
            raise RuntimeError(f"rollout token/logprob length mismatch: {len(new_ids)} vs {len(new_logps)}")
        new_text = rollout_result.completions[0]
        all_outputs.append(new_text)
        response_ids.extend(new_ids)
        response_mask.extend([int(t != tokenizer.eos_token_id) for t in new_ids])
        response_old_logps.extend(new_logps)
        final_context = context + new_text
        calls = parse_tool_calls(new_text)
        if not calls:
            break
        unfinished = turn == max_turns - 1
        assistant_message = {"role": "assistant", "content": new_text}
        messages.append(assistant_message)
        for call in calls:
            name, raw = call.get("name", ""), call.get("arguments", {})
            if isinstance(raw, str):
                try: raw = json.loads(raw)
                except: raw = {}
            result = execute_tool(name, raw) if isinstance(raw, dict) else None
            result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]  # 매우 큰 숫자가 토크나이저 메모리를 소진하는 것을 방지
            messages.append({"role": "tool", "content": result_str})

        marker = f"<|agent_observation_{id(messages)}_{len(response_ids)}|>"
        assistant_message["content"] += marker
        marked_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
        assistant_message["content"] = new_text
        _, found, observation = marked_context.partition(marker)
        if not found:
            raise RuntimeError("chat template did not preserve the assistant content boundary")
        obs_delta = tokenizer(observation, add_special_tokens=False)["input_ids"]
        if new_ids and new_ids[-1] == tokenizer.eos_token_id and obs_delta[:1] == [tokenizer.eos_token_id]:
            obs_delta = obs_delta[1:]
        observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))
        response_old_logps.extend([0.0] * len(obs_delta))
        final_context = observe_context

    final_output = all_outputs[-1] if all_outputs else ""
    prompt_ids = prompt_ids or []
    return final_output, final_context, prompt_ids, response_ids, response_mask, response_old_logps, list(all_outputs), unfinished

def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, num_gen, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda", workers=1):
    if workers > 1:
        jobs = [(messages, tools) for messages, tools in zip(messages_batch, tools_batch) for _ in range(num_gen)]
        def run(job):
            messages, tools = job
            return rollout_single(rollout_engine, tokenizer, [dict(m) for m in messages], tools,
                                  max_turns, max_new_tokens, thinking_ratio, device)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return tuple(map(list, zip(*pool.map(run, jobs))))
    all_completions = []
    all_contexts = []
    all_prompt_ids = []
    all_response_ids = []
    all_response_masks = []
    all_response_old_logps = []
    all_turn_outputs = []
    all_unfinished = []
    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):
            msgs_copy = [dict(m) for m in messages]
            completion, context, prompt_ids, response_ids, response_mask, response_old_logps, turn_outputs, unfinished = rollout_single(rollout_engine, tokenizer, msgs_copy, tools, max_turns, max_new_tokens, thinking_ratio, device)
            all_completions.append(completion)
            all_contexts.append(context)
            all_prompt_ids.append(prompt_ids)
            all_response_ids.append(response_ids)
            all_response_masks.append(response_mask)
            all_response_old_logps.append(response_old_logps)
            all_turn_outputs.append(turn_outputs)
            all_unfinished.append(unfinished)
    return all_completions, all_contexts, all_prompt_ids, all_response_ids, all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished

# ======== 보상 계산 ========
def validate_gt_in_text(text, gt_list):
    text = str(text)
    text_num = text.replace(',', '')
    nums = [float(x) for x in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?!\w|\.\d)', text_num)]
    matched = set()
    for gt in gt_list:
        value = str(gt).strip()
        numeric = value.replace(',', '')
        if re.fullmatch(r'[-+]?\d+(?:\.\d+)?', numeric):
            if any(abs(float(numeric) - number) < 1e-6 for number in nums):
                matched.add(gt)
        elif value and value.lower() in text.lower():
            matched.add(gt)
    return matched


def tool_task_reward(turns, calls, tools, gt, unfinished):
    """Score labeled tool tasks by successful execution and final answer agreement."""
    attempts = sum(max(t.count('<tool_call>'), t.count('</tool_call>')) for t in turns)
    if attempts != len(calls):
        return -1.5  # malformed JSON/tags must not fall back to generic text reward
    if not calls:
        return -1.0
    schemas = {t['function']['name']: t['function'] for t in tools}
    results = []
    for call in calls:
        name, args = call['name'], call.get('arguments', {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                return -1.5
        schema = schemas.get(name)
        if not schema or not isinstance(args, dict):
            return -1.5
        parameters = schema.get('parameters', {})
        if any(key not in args for key in parameters.get('required', [])):
            return -1.5
        for key, value in args.items():
            kind = parameters.get('properties', {}).get(key, {}).get('type')
            if kind == 'string' and not isinstance(value, str):
                return -1.5
            if kind == 'number' and (type(value) not in (int, float) or not math.isfinite(value)):
                return -1.5
        check = CHECK_ARGS.get(name)
        if not check or not check(args):
            return -1.5
        result = execute_tool(name, args)
        if result is None:
            return -1.5
        results.append(json.dumps(result, ensure_ascii=False))
    if unfinished:
        return -0.5
    final_text = turns[-1].split('</tool_call>')[-1].strip() if turns else ''
    matched = validate_gt_in_text(final_text, gt) & validate_gt_in_text('\n'.join(results), gt)
    fraction = len(matched) / len(set(gt)) if gt else 0.0
    score = 0.5 + 2.5 * fraction if matched else -0.5
    return max(-3.0, score - rep_penalty(final_text))


def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen, reward_model=None, device="cuda", turn_outputs_batch=None, unfinished_batch=None):
    rewards = torch.zeros(len(completions), device=device)
    for idx, response in enumerate(completions):
        reward, answer = 0.0, response
        sample_idx = idx // num_gen
        tools = tools_batch[sample_idx]
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch is not None else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch is not None else False
        turn_answers = [turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip() for turn in turn_outputs]
        answer = turn_answers[-1] if turn_answers else response.strip()
        valid_names = {t['function']['name'] for t in tools} if tools else set()
        tool_calls = []
        for turn_answer in turn_answers: tool_calls.extend(parse_tool_calls(turn_answer))  # 도구 호출 파싱
        # In agent_rl_en, tools plus nonempty gt identifies a labeled tool task.
        # Merely exposing tools does not require calling them on general dialogue.
        if tools and gt_batch[sample_idx]:
            rewards[idx] = tool_task_reward(turn_answers, tool_calls, tools, gt_batch[sample_idx], unfinished)
            continue
        reward -= 0.5 * sum(abs(turn.count('<tool_call>') - turn.count('</tool_call>')) for turn in turn_answers)  # 태그 불일치 감점
        # -------- 도구 호출 없음: 형식 및 보상 모델 점수 --------
        if not tool_calls:
            reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5  # 길이 점수
            if '</think>' in response:
                think, answer = response.split('</think>', 1)
                reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5  # 사고 길이 점수
                reward += 0.25 if response.count('</think>') == 1 else -0.25  # 사고 태그 닫힘 점수
                answer = answer.strip()
            if reward_model is not None:
                prompt = prompts[sample_idx]
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                score = reward_model.get_score(messages, answer)
                reward += score  # 보상 모델 점수
            reward -= rep_penalty(answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 전체 점수 클리핑
        # -------- 도구 호출 있음: 실행 결과 보상 --------
        else:
            gt = gt_batch[sample_idx]
            valid_call_count = 0
            for tool_call in tool_calls:
                name, raw = tool_call.get("name", ""), tool_call.get("arguments", {})
                if isinstance(raw, str):
                    try: raw = json.loads(raw)
                    except: raw = {}
                check = CHECK_ARGS.get(name)
                valid_call_count += int(bool(isinstance(raw, dict) and name in valid_names and check and check(raw)))
            tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)  # 도구 호출 수 차이
            reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap  # 도구 호출 일치 점수
            
            final_text = "" if unfinished else (answer.split('</tool_call>')[-1] if '</tool_call>' in answer else answer)
            verified = validate_gt_in_text(final_text, gt) if gt else set()
            if gt: reward += 2.5 * len(verified) / len(gt)  # 정답 일치 점수
            if unfinished: reward -= 0.5  # 미완료 감점
            reward -= rep_penalty(final_text if final_text else answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 전체 점수 클리핑
    return rewards

# ================================ 도구 및 Reward = End ================================
def backward_policy_microbatches(model, ref_model, input_ids, full_mask, completion_mask,
                                old_logps, advantages, args, lm_config, autocast_ctx):
    """Accumulate the same global response-mean objective with bounded activation memory."""
    count = input_ids.size(0)
    size = args.train_microbatch_size or count
    valid_count = (completion_mask.sum(1) > 0).sum().clamp(min=1)
    total_loss, kl_sum = 0.0, 0.0
    for start in range(0, count, size):
        stop = min(start + size, count)
        length = int(full_mask[start:stop].sum(1).max().item())
        ids = input_ids[start:stop, :length]
        attention = full_mask[start:stop, :length]
        mask = completion_mask[start:stop, :length-1]
        with autocast_ctx:
            result = model(ids, attention_mask=attention)
            aux = result.aux_loss if lm_config.use_moe else ids.new_tensor(0.0, dtype=torch.float32)
            logps = F.log_softmax(result.logits[:, :-1], dim=-1).gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            ref_logps = compute_per_token_logps(ref_model, ids, length-1, attention_mask=attention)
        delta = ref_logps - logps
        kl = torch.exp(delta) - delta - 1
        ratio = torch.exp(logps - old_logps[start:stop, :length-1])
        advantage = advantages[start:stop].unsqueeze(1)
        if args.loss_type == 'cispo':
            per_token_loss = -(torch.clamp(ratio, max=args.epsilon_high).detach() * advantage * logps - args.beta * kl)
        else:
            clipped = torch.clamp(ratio, 1-args.epsilon, 1+args.epsilon)
            per_token_loss = -(torch.min(ratio * advantage, clipped * advantage) - args.beta * kl)
        tokens = mask.sum(1)
        valid = tokens > 0
        policy_loss = ((per_token_loss * mask).sum(1)[valid] / tokens[valid].clamp(min=1)).sum() / valid_count
        loss = (policy_loss + aux * ((stop-start)/count)) / args.accumulation_steps
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite policy loss')
        loss.backward()
        total_loss += loss.detach().item() * args.accumulation_steps
        kl_sum += (delta.detach() * mask).sum().item()
    return total_loss, kl_sum / max(completion_mask.sum().item(), 1)


def rl_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model=None, start_step=0, wandb=None, use_sglang=False):
    for step, batch in enumerate(loader, start=start_step + 1):
        messages_batch = batch['messages']
        tools_batch = batch['tools']
        gt_batch = batch['gt']

        with torch.no_grad():
            completions, contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch, turn_outputs_batch, unfinished_batch = rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, args.num_generations, max_turns=3, max_new_tokens=args.max_gen_len, thinking_ratio=args.thinking_ratio, device=args.device, workers=args.rollout_workers)

        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, tools=t) for m, t in zip(messages_batch, tools_batch)]
        packed_samples = []
        for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
            ids = p + r
            mask = [0] * len(p) + m
            old_logps = [0.0] * max(len(p) - 1, 0) + old_lp
            if len(ids) > args.max_total_len:
                ids = ids[-args.max_total_len:]
                mask = mask[-args.max_total_len:]
                old_logps = old_logps[-(len(ids) - 1):]
            prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
            packed_samples.append((ids, mask, prompt_len, old_logps))
        seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=args.device)
        max_len = seq_lens.max().item()
        input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed_samples], device=args.device)
        prompt_lens = torch.tensor([prompt_len for _, _, prompt_len, _ in packed_samples], device=args.device)
        full_response_masks = torch.tensor([mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples], device=args.device, dtype=torch.float32)
        old_per_token_logps = torch.tensor([old_logps + [0.0] * ((max_len - 1) - len(old_logps)) for _, _, _, old_logps in packed_samples], device=args.device, dtype=torch.float32)
        full_mask = (torch.arange(max_len, device=args.device).unsqueeze(0) < seq_lens.unsqueeze(1)).long()

        rewards = calculate_rewards(prompts, completions, gt_batch, tools_batch, args.num_generations, reward_model, device=args.device, turn_outputs_batch=turn_outputs_batch, unfinished_batch=unfinished_batch)

        completion_mask = full_response_masks[:, 1:]
        is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
        eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1, device=args.device, dtype=torch.long)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        pos = torch.arange(completion_mask.size(1), device=args.device).unsqueeze(0)
        completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
        token_counts = completion_mask.sum(dim=1)
        valid_rows = token_counts > 0

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(messages_batch)):
                Logger(f"[DEBUG] step={step}, gt[{i}]: {repr(gt_batch[i])}")
                Logger('-'*100)
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    plen, slen = prompt_lens[idx].item(), seq_lens[idx].item()
                    Logger(f"{'=' * 30} [DEBUG] gen[{i}][{j}] CONTEXT_BEGIN {'=' * 30}")
                    Logger(contexts[idx])
                    Logger(f"{'=' * 31} [DEBUG] gen[{i}][{j}] CONTEXT_END {'=' * 31}")
                    Logger(f"[DEBUG] gen[{i}][{j}] prompt_len={plen}, seq_len={slen}")
                    tokens = input_ids[idx, plen:slen].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=False)
                    Logger(f"{'=' * 28} [DEBUG] gen[{i}][{j}] COMPLETION_BEGIN [{plen}:{slen}] {'=' * 28}")
                    Logger(text)
                    Logger(f"{'=' * 29} [DEBUG] gen[{i}][{j}] COMPLETION_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{i}][{j}] reward={rewards[idx].item():.4f}")
                    Logger('='*100)

        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        loss_value, kl_value = backward_policy_microbatches(model, ref_model, input_ids, full_mask,
            completion_mask, old_per_token_logps, advantages, args, lm_config, autocast_ctx)

        if step % args.accumulation_steps == 0 or step == iters:
            if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            pl = loss_value
            ar = rewards.mean().item()
            al = token_counts.float().mean().item()
            kl = kl_value
            gs = grouped_rewards.std(dim=1, unbiased=False).mean().item()
            am, ast = advantages.mean().item(), advantages.std().item()
            lr = optimizer.param_groups[0]['lr']
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), Reward:{ar:.4f}, KL:{kl:.4f}, GrpStd:{gs:.4f}, AdvStd:{ast:.4f}, Loss:{pl:.4f}, AvgLen:{al:.2f}, AdvMean:{am:.4f}, LR:{lr:.8f}')
            if wandb and is_main_process():
                wandb.log({"reward":ar,"kl_ref":kl,"group_reward_std":gs,"advantages_std":ast,"policy_loss":pl,"avg_response_len":al,"advantages_mean":am,"learning_rate":lr})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        if step % (args.policy_update_interval or args.save_interval) == 0 or step == iters: rollout_engine.update_policy(model)

        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Agent RL")
    parser.add_argument("--save_dir", type=str, default="../out", help="모델 저장 디렉토리")
    parser.add_argument('--save_weight', default='agent', type=str, help="저장할 가중치 이름")
    parser.add_argument("--epochs", type=int, default=1, help="학습 에포크 수")
    parser.add_argument("--batch_size", type=int, default=2, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="학습률")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="학습 장치")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="데이터 유형 bfloat16/float16")
    parser.add_argument("--num_workers", type=int, default=8, help="데이터 로딩 작업자 수")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="기울기 누적 단계 수")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="기울기 클리핑 임계값")
    parser.add_argument("--log_interval", type=int, default=1, help="로그 출력 간격")
    parser.add_argument("--save_interval", type=int, default=10, help="모델 저장 간격")
    parser.add_argument("--policy_update_interval", type=int, default=0, help="Rollout 가중치 갱신 간격(0이면 저장 간격 사용)")
    parser.add_argument('--hidden_size', default=768, type=int, help="모델 은닉층 차원")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="모델 층 수")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="MoE 사용 여부")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="최대 시퀀스 길이")
    parser.add_argument("--max_gen_len", type=int, default=768, help="한 번에 생성할 최대 길이")
    parser.add_argument("--max_total_len", type=int, default=2500, help="학습 시 최종 전체 길이 상한")
    parser.add_argument("--data_path", type=str, default="../dataset/agent_rl.jsonl", help="학습 데이터 경로")
    parser.add_argument("--num_generations", type=int, default=4, help="프롬프트당 생성 수")
    parser.add_argument("--beta", type=float, default=0.1, help="KL 발산 페널티 계수")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="손실 유형")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO의 PPO 클리핑 epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon 상한")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="불러올 사전 학습 가중치 이름")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="체크포인트에서 복원할지 여부")
    parser.add_argument("--use_wandb", action="store_true", help="wandb 기록 사용 여부")
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-RL", help="wandb 프로젝트 이름")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="torch.compile 사용 여부")
    parser.add_argument("--debug_mode", action="store_true", help="디버그 모드")
    parser.add_argument("--debug_interval", type=int, default=20, help="디버그 로그 간격")
    parser.add_argument("--thinking_ratio", type=float, default=0.1, help="확률에 따라 사고 모드 활성화(0.0~1.0)")
    parser.add_argument("--reward_model_path", type=str, default="Skywork/Skywork-Reward-V2-Qwen3-1.7B", help="보상 모델 경로")
    parser.add_argument("--reward_score_scale", type=float, default=10.0, help="원시 보상을 이 값으로 나눈 뒤 [-3,3] 클리핑")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout 엔진 유형")
    parser.add_argument("--rollout_workers", type=int, default=1, help="SGLang rollout 동시 요청 수")
    parser.add_argument("--train_microbatch_size", type=int, default=0, help="역전파를 나눌 응답 수(0이면 전체)")
    parser.add_argument("--tokenizer_path", default="../model")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang 서버 URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang 토크나이저 경로")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_agent", help="SGLang 공유 저장소 경로")
    args = parser.parse_args()
    if args.rollout_workers > 1 and args.rollout_engine != 'sglang':
        parser.error('--rollout_workers > 1 requires the sglang engine')

    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb = wandb.init(project=args.wandb_project, name=f"Agent-RL-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}", id=wandb_id, resume=resume, mode=args.wandb_mode, config=vars(args))

    model, tokenizer = init_model(lm_config, args.from_weight, tokenizer_path=args.tokenizer_path, save_dir=args.save_dir, device=args.device)

    ref_model, _ = init_model(lm_config, args.from_weight, tokenizer_path=args.tokenizer_path, save_dir=args.save_dir, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=dtype, score_scale=args.reward_score_scale)
    Logger(f'Loaded reward model from {args.reward_model_path}')
    # Rollout 엔진
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    def collate_fn(batch): return {'messages': [b['messages'] for b in batch], 'tools': [b['tools'] for b in batch], 'gt': [b['gt'] for b in batch]}
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        # train_ppo와 동일: RoPE 버퍼가 모든 rank에서 같으므로 매 단계 브로드캐스트할 필요 없음
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    rollout_engine.update_policy(model)

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        if skip > 0:
            Logger(f'Epoch [{epoch+1}/{args.epochs}]: skip {start_step} steps')
            rl_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            rl_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))

    if wandb:
        wandb.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
