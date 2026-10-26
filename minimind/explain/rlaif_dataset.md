# RLAIFDataset의 데이터 처리와 GRPO 생각 모드

이 문서는 두 가지 질문을 다룹니다.

1. 간단한 샘플 데이터가 `RLAIFDataset`에서 어떻게 처리되는가?
2. GRPO에서 모델에 단계별 추론을 요청하는 부분은 어디인가?

관련 코드:

- [dataset/lm_dataset.py](../dataset/lm_dataset.py): `RLAIFDataset`, `pre_processing_chat`
- [model/tokenizer_config.json](../model/tokenizer_config.json): 채팅 템플릿
- [trainer/train_grpo.py](../trainer/train_grpo.py): 데이터셋 생성, rollout, 보상 계산
- [GRPO loss 설명](grpo_loss.md)

## 1. RLAIFDataset의 역할

`RLAIFDataset`은 기존 대화의 마지막 답변을 제외하고, 모델이 새 답변을 생성할 **프롬프트 문자열**을 만듭니다. 실제 응답 생성과 보상 계산은 이후 GRPO 학습 코드에서 수행합니다.

### 1-1. 원본 데이터 읽기

데이터 한 건을 다음과 같이 가정합니다. JSONL 파일에는 객체 한 건을 한 줄에 저장합니다.

```json
{
  "conversations": [
    {"role": "user", "content": "1 + 1은?"},
    {"role": "assistant", "content": "2입니다."}
  ]
}
```

초기화할 때 `load_dataset('json', ...)`으로 데이터를 읽고, `__getitem__()`에서 샘플의 `conversations`를 가져옵니다.

### 1-2. system 메시지 전처리

```python
conversations = pre_processing_chat(conversations)
```

system 메시지가 없는 일반 대화에는 20% 확률로 준비된 system 메시지 중 하나를 앞에 붙입니다. 예를 들면 다음과 같습니다.

```json
[
  {"role": "system", "content": "You are a helpful AI assistant."},
  {"role": "user", "content": "1 + 1은?"},
  {"role": "assistant", "content": "2입니다."}
]
```

이미 system 메시지가 있으면 유지합니다. 메시지 중 참인 `tools` 값이 있으면 이 전처리 함수는 대화를 그대로 반환합니다.

이후 예시는 system 메시지가 추가되지 않은 경우로 설명합니다.

### 1-3. 마지막 답변 제외

채팅 템플릿에는 전체 대화 대신 다음 목록을 전달합니다.

```python
conversations[:-1]
```

샘플에서는 마지막 답변이 제외되고 질문만 남습니다.

```json
[
  {"role": "user", "content": "1 + 1은?"}
]
```

원본 답변 `"2입니다."`를 정답으로 제공하는 대신 모델이 새 응답을 생성하도록 준비합니다. 여러 턴 대화라면 마지막 메시지를 제외한 이전 대화는 남습니다.

코드는 마지막 메시지의 역할을 검사하지 않고 무조건 제외합니다. 따라서 이 사용 방식에서는 마지막 메시지가 assistant 답변인 데이터를 넣어야 합니다. 마지막 메시지가 사용자 질문이면 그 질문이 제거됩니다.

### 1-4. 생각 모드 선택

```python
use_thinking = random.random() < self.thinking_ratio
```

데이터를 가져올 때마다 확률적으로 생각 모드를 선택합니다.

| 설정 위치 | 기본 thinking_ratio |
|---|---:|
| `RLAIFDataset` 생성자 | 0.5 |
| `train_grpo.py` 실행 인자 | 0.9 |

GRPO 학습 코드는 실행 인자의 값을 데이터셋에 전달합니다. 기본 실행이라면 약 90% 확률로 생각 모드가 선택되며, 정확히 매 10건 중 9건이 선택된다는 의미는 아닙니다.

### 1-5. 채팅 템플릿 적용

```python
return self.tokenizer.apply_chat_template(
    conversations[:-1],
    tokenize=False,
    open_thinking=use_thinking,
    add_generation_prompt=True,
)
```

| 옵션 | 의미 |
|---|---|
| `tokenize=False` | 토큰 ID가 아닌 문자열 반환 |
| `open_thinking=use_thinking` | 템플릿에 생각 영역을 열어 둘지 전달 |
| `add_generation_prompt=True` | 응답 생성을 시작할 assistant 접두부 추가 |

로컬 `model/tokenizer_config.json` 기준으로 생각 모드가 켜지면 다음 문자열이 됩니다.

```text
<|im_start|>user
1 + 1은?<|im_end|>
<|im_start|>assistant
<think>
```

마지막 `<think>` 뒤의 줄바꿈까지 프롬프트에 포함됩니다. 모델은 그 뒤를 이어서 생성합니다.

생각 모드가 꺼지면 빈 생각 영역을 닫아 둡니다.

```text
<|im_start|>user
1 + 1은?<|im_end|>
<|im_start|>assistant
<think>

</think>

```

정확한 끝부분은 `<think>\n\n</think>\n\n`입니다. 이 클래스는 빈 생각 영역을 제거하는 `post_processing_chat()`을 호출하지 않습니다.

템플릿의 동작은 전달한 토크나이저에 달려 있습니다. 다른 템플릿을 사용하면 `open_thinking` 옵션이나 출력 형식이 다를 수 있습니다.

### 1-6. 최종 반환값

생각 모드가 켜진 샘플의 반환값은 다음과 같습니다.

```python
{
    "prompt": (
        "<|im_start|>user\n"
        "1 + 1은?<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
    ),
    "answer": "",
}
```

`answer`는 항상 빈 문자열입니다. 원본 답변을 담지 않으며, GRPO 학습 루프는 `batch['prompt']`를 사용합니다.

`RLAIFDataset`은 `x`, `y`, 손실 마스크를 만들지 않습니다. `max_length`도 저장만 하고 프롬프트 길이 제한에 사용하지 않습니다. 프롬프트 토큰화, 패딩과 길이 제한은 GRPO 학습 코드에서 수행합니다.

### 1-7. 이후 GRPO 처리와 연결

```text
원본 대화
  → 마지막 답변 제외
  → 생각 모드를 선택하고 프롬프트 문자열 생성
  → GRPO 코드에서 토큰화·패딩·길이 제한
  → rollout 엔진이 같은 질문에 응답 여러 개 생성
  → 각 응답의 보상 계산
  → 그룹 내 advantage 계산
  → GRPO loss 계산
```

[DPODataset](dpo_dataset.md)은 데이터에 있는 선호·비선호 응답을 학습용 텐서로 변환합니다. 반면 `RLAIFDataset`은 새 응답을 생성할 프롬프트를 준비합니다.

## 2. “step by step으로 추론하라”는 요청은 어디에 있는가?

현재 확인한 GRPO 경로에는 “step by step으로 추론하라”는 문장을 직접 추가하는 부분이 없습니다. 대신 **생각 영역을 여는 프롬프트와 생성 응답의 형식 보상**으로 생각 과정을 작성하는 형식을 유도합니다. 입력 데이터 자체에 별도의 추론 지시가 있는지는 각 데이터를 확인해야 합니다.

### 2-1. RLAIFDataset에서 생각 모드 전달

`dataset/lm_dataset.py`의 `RLAIFDataset.create_chat_prompt()`가 다음 옵션을 전달합니다.

```python
open_thinking=use_thinking
```

이것은 자연어 지시가 아니라 채팅 템플릿에서 사용하는 옵션입니다.

### 2-2. 채팅 템플릿에서 열린 think 태그 추가

`model/tokenizer_config.json`의 템플릿은 `add_generation_prompt=True`이고 `open_thinking=True`일 때 프롬프트를 다음 접두부로 끝냅니다.

```text
<|im_start|>assistant
<think>
```

이 형식을 학습한 모델이라면 생각 영역을 작성하고 `</think>`로 닫은 뒤 답변을 생성하는 패턴을 따를 수 있습니다. 태그를 추가하는 것만으로 단계별 추론 능력이나 정확성이 보장되지는 않습니다.

### 2-3. GRPO 보상 함수에서 길이와 종료 태그 평가

`trainer/train_grpo.py`의 `calculate_rewards()`에는 다음 처리가 있습니다.

```python
if '</think>' in response:
    thinking_content, answer_content = response.split('</think>', 1)
    rewards[response_idx] += (
        1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
    )
    rewards[response_idx] += (
        0.25 if response.count('</think>') == 1 else -0.25
    )
    answer = answer_content.strip()
```

여기서 `response`는 생성된 응답입니다. 생각 모드에서 여는 `<think>`는 이미 프롬프트에 들어 있고, 모델이 그 뒤의 내용과 닫는 태그를 생성하는 형태입니다.

| 조건 | 추가 점수 |
|---|---:|
| 첫 `</think>` 앞 내용의 공백 제거 후 길이가 20~300자 | +1.0 |
| 위 길이 조건을 만족하지 않음 | -0.5 |
| `</think>`가 정확히 한 번 등장 | +0.25 |
| `</think>`가 여러 번 등장 | -0.25 |

이 길이는 토큰 수가 아니라 파이썬 문자열의 `len()` 기준입니다. 응답에 `</think>`가 아예 없으면 이 조건문 내부의 보상과 페널티는 적용되지 않습니다.

생성된 응답 전체의 길이 보상, 반복 페널티, 보상 모델의 점수도 별도로 반영됩니다. `</think>`가 있으면 이후 부분을 `answer`로 사용해 반복 페널티와 보상 모델 점수를 계산합니다.

이 규칙은 생각 영역의 **길이와 종료 태그 개수**를 검사합니다. 단계별 논리가 맞는지, 추론이 답을 올바르게 뒷받침하는지를 직접 검증하는 코드는 아닙니다.

## 3. 수강생이 구분해야 할 세 단계

| 단계 | 하는 일 | 보장하지 않는 것 |
|---|---|---|
| 데이터셋 | 응답 생성을 위한 프롬프트 준비 | 정답 생성 |
| 생각 태그 | 학습된 생각 형식으로 이어 쓰도록 유도 | 논리적으로 올바른 단계별 추론 |
| 보상과 GRPO loss | 점수가 높은 생성 응답을 선호하도록 학습 | 형식 점수만으로 추론 품질 검증 |

이 문서는 코드와 로컬 템플릿을 바탕으로 한 처리 과정 설명입니다. 실제 모델을 실행해 생성 결과나 추론 능력을 검증한 결과는 아닙니다.
