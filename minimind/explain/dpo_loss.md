# 샘플 데이터로 이해하는 DPO loss

DPO는 같은 질문에 대한 선호 응답(chosen)과 비선호 응답(rejected)을 비교해 정책 모델을 학습합니다. 고정된 참조 모델을 기준으로, 정책 모델의 chosen/rejected 확률 비율이 커질수록 손실이 작아집니다.

관련 코드: [exercise/dpo_example.py](../exercise/dpo_example.py)의 3번 셀.

## 1. 샘플 데이터

- 질문: **“1 + 1은?”**
- chosen(선호 응답): **“2”**
- rejected(비선호 응답): **“3”**

각 응답이 토큰 하나라고 가정하고, 질문이 주어졌을 때 모델이 응답에 부여하는 확률을 다음과 같이 설정합니다.

| 모델 | chosen `"2"` | rejected `"3"` |
|---|---:|---:|
| 참조 모델 — 고정된 기준 | 0.25 | 0.25 |
| 정책 모델 — 학습할 모델 | 0.6 | 0.2 |

나머지 확률은 다른 토큰에 배정되어 있습니다. 여기서는 두 응답의 확률만 비교합니다. 아래에서 `log`는 자연로그입니다.

## 2. 각 모델의 선호 정도 계산

chosen과 rejected의 로그 확률 차이를 계산합니다. 로그의 성질에 따라 이는 두 확률 비율의 로그와 같습니다.

```text
정책 모델의 차이 = log(0.6) − log(0.2)
                 = log(3)
                 ≈ 1.0986

참조 모델의 차이 = log(0.25) − log(0.25)
                 = log(1)
                 = 0
```

정책 모델은 chosen에 rejected보다 3배 높은 확률을 주고, 참조 모델은 동일한 확률을 줍니다.

## 3. 참조 모델 대비 마진 계산

정책 모델의 선호가 참조 모델보다 얼마나 강한지 계산합니다.

```text
margin = 정책 모델의 로그 확률 차이 − 참조 모델의 로그 확률 차이
       = 1.0986 − 0
       = 1.0986
```

## 4. β를 곱하고 sigmoid 적용

`β`는 마진이 손실에 반영되는 크기를 조절합니다. 예제에서는 `β = 0.1`입니다.

```text
β × margin = 0.1 × 1.0986 = 0.10986

sigmoid(x) = 1 / (1 + exp(−x))
sigmoid(0.10986) ≈ 0.5274
```

## 5. 음의 로그를 취해 DPO loss 계산

```text
loss = −log(sigmoid(β × margin))
     ≈ −log(0.5274)
     ≈ 0.6397
```

두 모델의 chosen/rejected 확률 비율이 같으면 마진은 0이고, 손실은 `−log(0.5) ≈ 0.6931`입니다. 이번에는 정책 모델이 참조 모델보다 chosen을 더 선호하므로 손실이 약 `0.6397`로 낮아집니다.

실제 학습에서는 이 손실이 줄어드는 방향으로 정책 모델을 업데이트합니다. 참조 모델은 고정합니다.

## 6. 파이썬으로 계산하기

다음 코드는 표준 라이브러리만으로 위 샘플의 손실을 계산합니다.

```python
import math

policy_chosen = 0.6
policy_rejected = 0.2
ref_chosen = 0.25
ref_rejected = 0.25
beta = 0.1

policy_gap = math.log(policy_chosen) - math.log(policy_rejected)
ref_gap = math.log(ref_chosen) - math.log(ref_rejected)
margin = policy_gap - ref_gap

sigmoid_value = 1 / (1 + math.exp(-beta * margin))
loss = -math.log(sigmoid_value)

print(f"DPO loss: {loss:.4f}")  # DPO loss: 0.6397
```

## 7. 응답이 여러 토큰인 경우

응답에 포함된 각 토큰의 로그 확률을 더해서 응답 전체의 로그 확률을 구합니다. 각 토큰의 확률은 질문과 앞선 응답 토큰이 주어진 조건에서 계산합니다.

```text
log P(응답 | 질문)
  = log P(첫 토큰 | 질문)
  + log P(둘째 토큰 | 질문, 첫 토큰)
  + …
```

예제 코드에서는 다음과 같이 계산합니다.

```python
ref_scores = (ref_log_probs * mask).sum(dim=1)
policy_scores = (policy_log_probs * mask).sum(dim=1)
```

`mask`는 응답 토큰 자리만 1이고 질문과 패딩 자리는 0입니다. 따라서 질문과 패딩의 로그 확률은 응답 점수에 포함하지 않습니다. 이렇게 구한 chosen과 rejected의 점수로 동일한 DPO loss 계산을 진행합니다.
