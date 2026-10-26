# 샘플 데이터로 이해하는 DPODataset

`DPODataset`은 선호 응답(chosen)과 비선호 응답(rejected)이 포함된 대화를 읽고, 각각을 모델 입력 `x`, 예측 대상 `y`, 손실 마스크 `mask`로 변환합니다. 확률과 DPO loss는 이후 모델과 학습 코드에서 계산합니다.

- 구현: [dataset/lm_dataset.py](../dataset/lm_dataset.py)의 `DPODataset`
- 채팅 템플릿: [model/tokenizer_config.json](../model/tokenizer_config.json)
- 학습 코드: [trainer/train_dpo.py](../trainer/train_dpo.py)
- 이어지는 설명: [DPO loss 계산](dpo_loss.md)

## 1. 원본 데이터 읽기

데이터 한 건은 다음과 같습니다. JSONL 파일에서는 이 객체 한 건을 한 줄에 저장합니다.

```json
{
  "chosen": [
    {"role": "user", "content": "1 + 1은?"},
    {"role": "assistant", "content": "2"}
  ],
  "rejected": [
    {"role": "user", "content": "1 + 1은?"},
    {"role": "assistant", "content": "3"}
  ]
}
```

두 필드 모두 질문과 응답을 포함한 대화입니다. `load_dataset('json', ...)`으로 데이터를 읽고, `__getitem__()`에서 한 쌍을 가져옵니다.

## 2. 채팅 템플릿 적용

각 대화에 다음 처리를 적용합니다.

```python
chosen_prompt = tokenizer.apply_chat_template(
    chosen, tokenize=False, add_generation_prompt=False
)
```

`tokenize=False`이므로 이 단계의 결과는 토큰 ID가 아닌 문자열입니다. 이미 assistant 응답이 있으므로 `add_generation_prompt=False`로 새 assistant 시작 표시를 덧붙이지 않습니다.

위에서 연결한 로컬 템플릿을 적용한 chosen 문자열은 다음과 같습니다.

```text
<|im_start|>user
1 + 1은?<|im_end|>
<|im_start|>assistant
<think>

</think>

2<|im_end|>
```

rejected 문자열은 다음과 같습니다.

```text
<|im_start|>user
1 + 1은?<|im_end|>
<|im_start|>assistant
<think>

</think>

3<|im_end|>
```

각 문자열은 마지막 `<|im_end|>` 뒤에도 줄바꿈을 포함합니다.

| 표시 | 의미 |
|---|---|
| `<|im_start|>` | 발화 시작 |
| `user`, `assistant` | 발화자의 역할 |
| `<|im_end|>` | 발화 종료 |
| `<think>…</think>` | 생각 과정 영역. 이 샘플에서는 비어 있음 |

실제로 적용되는 형식은 `DPODataset`에 전달한 토크나이저의 템플릿에 따라 달라집니다.

## 3. 빈 생각 영역 전처리

`post_processing_chat()`은 정확히 `<think>\n\n</think>\n\n` 형태인 빈 생각 영역을 80% 확률로 제거합니다. chosen과 rejected에 같은 제거 여부를 적용합니다.

제거된 chosen:

```text
<|im_start|>user
1 + 1은?<|im_end|>
<|im_start|>assistant
2<|im_end|>
```

제거된 rejected:

```text
<|im_start|>user
1 + 1은?<|im_end|>
<|im_start|>assistant
3<|im_end|>
```

내용이 있는 생각 영역을 모두 삭제하는 처리는 아닙니다.

## 4. 토큰화, 길이 제한과 패딩

전처리한 문자열을 토큰 ID로 바꿉니다.

```python
chosen_encoding = tokenizer(
    chosen_prompt,
    truncation=True,
    max_length=self.max_length,
    padding='max_length',
)
```

길면 `max_length`까지 자르고, 짧으면 패딩으로 길이를 맞춥니다.

이후 과정을 쉽게 보기 위해 다음 기호를 사용하겠습니다. **설명용 가상 토큰이며, 실제 토큰 ID나 실제 토큰 개수와는 다릅니다.** 실제 질문이나 역할 표시는 여러 토큰으로 나뉠 수 있습니다.

| 기호 | 단순화한 의미 |
|---|---|
| `U` | 사용자 시작 표시 |
| `Q` | 질문 `1 + 1은?` |
| `A` | assistant 시작 표시 |
| `E` | 발화 종료 표시와 뒤의 줄바꿈 |
| `2`, `3` | 응답 토큰 |
| `PAD` | 패딩 |

빈 생각 영역을 제거했고, 가상의 `max_length=8`이라고 하면:

```text
chosen:   [U, Q, E, A, 2, E, PAD, PAD]
rejected: [U, Q, E, A, 3, E, PAD, PAD]
```

## 5. 응답 부분에 손실 마스크 설정

`generate_loss_mask()`는 assistant 시작 토큰열 뒤부터 종료 토큰열까지 1로 표시합니다. 실제 코드가 찾는 시작 토큰열은 `bos_token + 'assistant\n'`, 종료 토큰열은 `eos_token + '\n'`을 토큰화한 결과입니다.

```text
토큰: [U, Q, E, A, 2, E, PAD, PAD]
mask: [0, 0, 0, 0, 1, 1,   0,   0]
```

실제 문자열의 구간으로 보면 다음과 같습니다.

```text
<|im_start|>user\n        → mask 0
1 + 1은?<|im_end|>\n      → mask 0
<|im_start|>assistant\n   → mask 0
2<|im_end|>\n             → mask 1
뒤에 붙은 패딩            → mask 0
```

응답 내용뿐 아니라 종료 표시와 뒤의 줄바꿈도 학습 대상입니다. 빈 생각 영역이 유지되면 그 영역도 assistant 응답에 포함되어 마스크가 1이 됩니다. 여러 assistant 발화가 있으면 해당 구간들을 모두 표시합니다.

위 예시는 정상적으로 종료된 대화 기준입니다. 이 구현은 종료 토큰열을 찾지 못하면 배열 끝까지 마스크를 표시하며, 잘림으로 응답이 사라지는 경우를 별도로 걸러내지는 않습니다.

## 6. 한 칸 이동해 입력과 예측 대상 만들기

모델이 다음 토큰을 예측하도록 배열을 나눕니다.

```python
x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
```

- `x`: 마지막 토큰을 제외한 입력
- `y`: 첫 토큰을 제외한 예측 대상
- `mask`: `y`와 같은 위치로 이동한 손실 마스크

```text
x:    [U, Q, E, A, 2, E, PAD]
y:    [Q, E, A, 2, E, PAD, PAD]
mask: [0, 0, 0, 1, 1,   0,   0]
```

주요 예측은 다음과 같습니다. 모델은 해당 자리까지의 앞선 입력을 함께 봅니다.

| 지금까지 읽은 입력 | 예측 대상 | 손실 포함 |
|---|---|---|
| `U Q E A` | `2` | 예 |
| `U Q E A 2` | `E` | 예 |

마스크도 한 칸 이동해야 입력 위치가 아니라 **예측 대상 위치**에 맞춰 손실을 계산할 수 있습니다.

## 7. 최종 반환값

동일한 처리를 rejected에도 적용해 텐서 6개를 반환합니다.

```text
x_chosen:      [U, Q, E, A, 2, E, PAD]
y_chosen:      [Q, E, A, 2, E, PAD, PAD]
mask_chosen:   [0, 0, 0, 1, 1,   0,   0]

x_rejected:    [U, Q, E, A, 3, E, PAD]
y_rejected:    [Q, E, A, 3, E, PAD, PAD]
mask_rejected: [0, 0, 0, 1, 1,   0,   0]
```

실제로는 토큰 ID 또는 0·1로 구성된 `torch.long` 텐서이며, 각 텐서의 길이는 `max_length - 1`입니다.

## 8. DPO loss와 연결

학습 코드는 chosen 배치 뒤에 rejected 배치를 이어 붙인 뒤 다음 순서로 처리합니다.

1. `x`를 정책 모델과 참조 모델에 넣어 각 위치의 logits를 구합니다.
2. `y`에 해당하는 대상 토큰의 로그 확률을 꺼냅니다.
3. `mask`를 곱하고 응답 구간의 로그 확률을 합산합니다.
4. 두 모델의 chosen/rejected 점수 차이를 비교해 DPO loss를 계산합니다.

```python
ref_scores = (ref_log_probs * mask).sum(dim=1)
policy_scores = (policy_log_probs * mask).sum(dim=1)
```

[DPO loss 설명](dpo_loss.md)의 단순한 수치 예제는 응답 토큰 하나의 확률로 계산합니다. 실제 `DPODataset`에서 만들어진 마스크는 종료 토큰열도 포함하므로, 실제 학습 점수에는 해당 토큰들의 로그 확률도 더해집니다.
