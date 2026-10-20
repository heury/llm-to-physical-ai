# MiniMind-J · Parallel Typed Decisions

Runpod에서 실제 `LocalLLaMA/typed-decisions`로 학습하는 별도 실행기는 [scripts/runpod_typed/README.md](scripts/runpod_typed/README.md)에 설명했습니다. 질문별 rubric과 soft gold 분포를 유지하며, 공식 test와 분리한 검증 데이터로 최적 epoch를 선택합니다.

공통 입력 하나에 대해 **여러 질문을 MiniMind forward 한 번으로 처리**하는 교육용 모델입니다. 질문 타입에 따라 `choice`는 문자열, `score`는 정수, `noul`은 boolean으로 답합니다. 별도 decision MLP 없이 **기존 LM head**를 재사용합니다.

`noul`은 참고한 Unsloth 코드의 boolean 타입 이름입니다. JSON `null`과 다릅니다.

## 빠른 실행

Python 3.11 이상. `minimind-j` 폴더에서 실행하세요. 기본은 CPU이며 모델 다운로드가 필요 없습니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py --epochs 30 --validation data/validation.jsonl
python decide.py --checkpoint out/decision --request data/request.json
python -m pytest -q
```

예제 요청은 다음 세 질문을 한 번에 처리합니다.

```json
{
  "input": "배송이 언제 오나요?",
  "questions": [
    {"id": "route", "type": "choice", "question": "문의 분류?",
     "choices": ["배송 문의", "환불 요청", "계정 문의"]},
    {"id": "urgency", "type": "score", "question": "긴급도? 일반0, 지연1, 환불2",
     "min": 0, "max": 2},
    {"id": "refund", "type": "noul", "question": "환불 또는 취소 요청?"}
  ]
}
```

출력 예시의 `answers` 값은 각각 string / integer / boolean입니다. `details`에는 각 질문의 허용 값과 그 순서에 대응하는 확률을 출력합니다.

```json
{
  "answers": {"route": "배송 문의", "urgency": 0, "refund": false}
}
```

## 질문 타입과 학습 데이터

JSONL 한 줄에 입력 하나와 질문 목록을 넣습니다. 질문 `id`는 입력 안에서 유일해야 합니다. 학습 시 각 질문에 `answer`를 추가합니다. 추론에서는 생략합니다.

| type | 질문 설정 | answer / 출력 타입 |
| --- | --- | --- |
| `choice` | `choices`: 중복 없는 후보 문자열 목록 | 목록의 문자열 자체. 인덱스가 아님 |
| `score` | `min`, `max`: 포함 범위의 정수 | 범위 내 정수 |
| `noul` | 별도 범위 없음 | JSON `true` 또는 `false` |

```json
{"input":"환불을 요청합니다","questions":[{"id":"route","type":"choice","question":"문의 분류?","choices":["배송 문의","환불 요청"],"answer":"환불 요청"},{"id":"urgent","type":"score","question":"긴급도? 일반0, 지연1, 환불2","min":0,"max":2,"answer":2},{"id":"refund","type":"noul","question":"환불 또는 취소 요청?","answer":true}]}
```

`score`는 **유한한 정수 등급 분류**입니다. 연속값 회귀가 아니며, argmax 등급을 반환합니다. 등급 간 거리 손실은 추가하지 않고 세 타입 모두 cross entropy로 학습합니다. boolean 정답에 `0`, `1`, 문자열 `"true"`를 쓰거나 점수에 `true`를 넣으면 오류를 냅니다.

학습 CLI의 dataset은 모든 질문에 정답이 있어야 합니다. Python collator/model API는 일부 질문만 정답이 있는 batch도 지원합니다. 정답 없는 질문은 loss에서 제외하고 전부 정답이 없으면 `loss=None`입니다.

구 버전의 `input/choices/label` 단일 선택 JSONL과 다음 CLI는 하나의 `choice` 질문으로 변환됩니다. `label`은 이 편의 형식에서만 0부터 시작하는 정수 인덱스입니다.

```bash
python decide.py --checkpoint out/decision \
  --input '배송이 언제 오나요?' --choices '배송 문의' '환불 요청' '계정 문의'
```

이 경우에도 출력은 새 구조의 `answers.answer`와 `details.answer`입니다. 구 형식 예제는 `data/train_single_choice.jsonl`에 보관했습니다.

## 한 번의 forward로 처리하는 원리

기존 v1은 후보마다 입력을 복제했습니다. v2는 **record 하나를 시퀀스 하나로** 구성합니다.

```text
공통 prefix: [BOS] 입력 [SEP]
branch 1: 질문 1 + 후보 목록 + Answer:
branch 2: 질문 2 + 후보 목록 + Answer:
branch 3: 질문 3 + 후보 목록 + Answer:
```

이 전체를 MiniMind에 한 번 전달합니다. 별도의 답변 텍스트를 autoregressive하게 생성하지 않습니다. 각 branch의 마지막 `Answer:` 토큰 hidden state에서 기존 LM head의 `A`, `B`, … 행을 이용해 점수를 동시에 읽습니다.

예를 들어 `choice` 후보 세 개는 `A/B/C`, score 0–2도 `A/B/C`, noul의 `false/true`는 `A/B`에 대응합니다. 후보와 코드의 대응은 질문 텍스트 안에 포함합니다. LM이 코드를 예측하면 Python에서 원래 JSON 타입으로 복원합니다.

```python
hidden = backbone(packed_input, block_attention_mask, local_positions)  # 1회
h = hidden[question_batch_indices, question_answer_positions]           # 모든 질문
scores = (h[:, None] * lm_head.weight[answer_token_ids]).sum(-1)
scores = scores.masked_fill(~option_mask, float("-inf"))
probabilities = softmax(scores, dim=-1)
loss = cross_entropy(scores, gold_code_indices)
```

코드는 LM head 전체 vocabulary를 계산한 뒤 모아오는 것과 같은 선형 계산이며, 실제 구현은 필요한 코드의 projection 행만 사용합니다. 새로 학습하는 decision head는 없습니다. 기본 설정에서는 LM head와 입력 embedding도 공유합니다.

### 질문끼리 간섭하지 않게 하는 두 가지 장치

1. **Block causal attention:** 공통 입력은 공통 입력만 보고, 각 질문은 공통 입력과 자기 branch의 앞부분만 봅니다. 다른 질문이나 다른 질문의 정답은 볼 수 없습니다.
2. **질문별 RoPE 위치:** 각 branch의 position을 공통 prefix 다음부터 다시 셉니다. 질문의 나열 순서나 앞 질문의 길이가 달라져도 다른 질문의 positional encoding은 변하지 않습니다.

따라서 eval 모드에서 질문을 섞거나 독립적으로 처리해도 해당 질문의 결과가 유지됩니다. 정답 `answer`는 입력 텍스트에 넣지 않습니다. **한 질문 안에서 후보 순서를 바꾸는 것은 답변 코드와 prompt를 바꾸므로 후보 순열 불변성까지 보장하는 구조는 아닙니다.**

공통 입력의 hidden state와 K/V는 각 layer에서 한 번 계산됩니다. 다만 이 교육용 구현은 dense attention tensor와 명시적인 mask를 사용하므로 전체 packed 길이 L에 대해 mask/attention 비용이 O(L²)입니다. 한 번의 호출이라는 의미가 질문 수와 무관한 일정 시간이나 sparse kernel의 속도를 보장하지는 않습니다. 새 질문을 나중에 추가할 때는 다시 forward해야 하며, packed 경로의 KV cache 재사용은 지원하지 않습니다.

## 크기와 제한

- 기본 모델: hidden 64, layers 2, query heads 4, KV heads 2, vocab 260.
- 오프라인 UTF-8 byte tokenizer 사용. 한국어 한 글자는 여러 토큰을 사용합니다.
- 기본 `--max-length 512`는 **공통 입력 + 모든 질문 branch를 합한 packed 토큰 수**입니다. 길이 초과는 오류이며 조용히 자르지 않습니다. 필요하면 `--max-length 1024`를 사용하세요.
- `choice` 후보와 `score` 단계는 각각 최대 26개(A–Z)입니다. 토크나이저가 사용할 각 코드 문자를 서로 다른 단일 토큰으로 표현하는지 검사합니다.
- 확률은 해당 질문의 후보 집합 안에서 정규화한 상대 확률이며, 보정된 신뢰도를 의미하지 않습니다.
- 기본 데이터는 훈련 12 records / 36 questions, 검증 3 records / 9 questions입니다. 교육용 예제로, 일반적인 의미 판단 능력을 입증하는 벤치마크가 아닙니다.

## 기존 MiniMind LM 사용

기본 모델은 무작위 초기화부터 학습합니다. 사전학습된 **현재 로컬 MiniMind와 동일한 dense 구조**의 native `.pth`와 정확한 config, tokenizer가 있으면 backbone과 **기존 LM head를 모두** 가져올 수 있습니다.

```bash
python train.py \
  --data data/train.jsonl \
  --lm-checkpoint /path/to/full_sft_768.pth \
  --lm-config /path/to/lm_config.json \
  --tokenizer /path/to/minimind/model \
  --freeze-backbone --lr 0.001 --epochs 10 \
  --output out/from-lm
```

`lm_config.json`은 원본 학습 설정의 `MiniMindConfig.to_dict()`여야 합니다. layers, vocab, heads, KV heads, intermediate size, head_dim과 RoPE 설정까지 맞아야 합니다. tokenizer 역시 LM 학습에 사용한 동일 tokenizer가 필요합니다. vocab 크기만으로 그 의미나 RoPE 설정까지 확인할 수는 없습니다. 모든 parameter는 strict load합니다.

`--freeze-backbone`은 LM head만 학습합니다. 입력 embedding과 LM head가 묶여 있으면 출력 weight를 복사해 분리한 뒤 backbone을 동결합니다. 따라서 입력 embedding을 포함한 backbone 가중치가 변하지 않습니다. 이때는 큰 원본 vocab에 해당하는 출력 projection 메모리가 추가됩니다. 이 변경도 checkpoint config에 저장합니다.

## 저장과 호환성

`out/decision/`에 `decision.pth`, `decision_config.json`, `training.json`을 저장합니다. HF tokenizer는 `tokenizer/`에도 저장합니다. 이 폴더와 `minimind-j` 코드만으로 추론하며 형제 프로젝트 import는 필요 없습니다. optimizer를 저장하는 학습 재개 기능과 best epoch 선택은 생략했습니다.

**v1의 scalar decision head 체크포인트는 v2와 호환되지 않습니다.** 구 파일을 로드하면 재학습 안내 오류를 냅니다. v2를 새로 학습하거나 원래 MiniMind LM에서 시작하세요. 기존 v1 checkpoint를 변환한 것처럼 취급하지 않습니다. LoRA adapter, MoE, Unsloth/CLEF checkpoint 변환도 지원 범위 밖입니다.

## 코드 읽기 순서

| 파일 | 주요 내용 |
| --- | --- |
| `dataset.py` | 타입 검증, 코드 매핑, 공통 입력 packing, block mask, local position ids |
| `model/model_minimind.py` | MiniMind backbone과 4D bool mask / batched RoPE 위치 지원 |
| `model/decision.py` | 기존 LM head의 질문별 점수, CE, typed decoding, native LM 변환, 저장/복원 |
| `train.py` | 질문 단위 loss/accuracy, AdamW, clipping, backbone 동결 |
| `decide.py` | JSON 요청으로 여러 질문을 동시에 판정 |
| `tests/test_decision.py` | 실제 학습, 한 번의 backbone 호출, 질문 간 격리, 순서/배치 불변성, 체크포인트 |

참고: [clef.py](https://github.com/unslothai/unsloth/blob/main/unsloth/models/clef.py), [decision.py](https://github.com/unslothai/unsloth/blob/main/unsloth/models/decision.py), [decision_from_lm.py](https://github.com/unslothai/unsloth/blob/main/unsloth/models/decision_from_lm.py), [decision_datasets.py](https://github.com/unslothai/unsloth/blob/main/unsloth/models/decision_datasets.py).

CLEF의 typed schema 개념을 참고했지만 joint schema head, 질문 간 상호작용, soft targets, calibration, ordinal distance loss, multimodal memory, custom kernel은 구현하지 않았습니다. Apache-2.0이며 출처는 `NOTICE`, 실제 검증 수치는 [docs/verification.md](docs/verification.md)에 기록합니다.

## 질문 표현·후보 순서 변화 평가

기존 체크포인트를 학습 없이 평가할 수 있습니다.

```bash
python scripts/evaluate_robustness.py \
  --checkpoint out/decision --data data/validation.jsonl \
  --paraphrases scripts/example_paraphrases.json \
  --output out/robustness.json
```

원본, 질문 표현 변경, choice 후보 역순을 각각 비교합니다. 후보의 실제 값으로
확률을 정렬해 답변 변경률·분포 차이를 계산하고, 정확도·KL도 함께 보고합니다.
동률에서는 실제 decoder의 첫 argmax 선택을 따릅니다. 질문 표현은 JSON의 명시적인
대응표로 관리하며 rubric과 정답은 보존합니다. 알 수 없는 지시문은 오류로 처리합니다.
Typed Decisions용 20개 지시문 대응표와 실행법은
[scripts/runpod_typed/README.md](scripts/runpod_typed/README.md)에 있습니다.

[로컬 평가 기록](docs/robustness-verification.md)은 작은 예제 모델의 결과입니다.
64M 모델의 강건성 결과와 구분해야 합니다.
