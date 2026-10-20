# v2: Parallel Typed Questions 검증

2026-10-20. Python 3.11.11 / PyTorch 2.14.1 / Transformers 5.20.2 / pytest 9.1.1 / macOS arm64 CPU.

## 자동 테스트

`python -m pytest -q`: **31 passed**.

- forward hook으로 **backbone 호출 1회**, input batch 크기가 후보/질문 수가 아닌 record 수인지 확인.
- choice/score/noul 질문 4개를 동시에 처리하고, 없는 후보의 확률 0과 확률 합 1 확인.
- 질문을 뒤집거나, 다른 질문의 길이를 바꾸거나, 질문 하나만 따로 처리해도 eval 결과 일치.
- 배치 padding, SDPA/수동 attention 경로의 점수 일치.
- 질문 하나일 때 기존 일반 causal MiniMind + 전체 LM head projection과 결과 일치.
- label 변경이 input token에 영향을 주지 않는지 확인. 일부/전체 미라벨 질문 loss 처리 확인.
- 실제 optimizer 학습으로 문자열 / 정수 / boolean 정답을 모두 학습.
- byte/HF tokenizer와 tied/untied LM head 저장·재로드 후 logit 일치.
- 원본 native LM weights strict 로딩, head-only 학습에서 backbone 가중치 bitwise 불변.
- score 범위/타입, noul boolean 타입, choice 값, 중복 id, 길이 초과 검증.
- scalar head를 사용하던 format v1을 명확한 재학습 오류로 거부.

## CPU 학습 → 저장 → 다중 질문 추론

```bash
python train.py --epochs 30 --validation data/validation.jsonl \
  --output /tmp/minimind-j-v2-smoke
python decide.py --checkpoint /tmp/minimind-j-v2-smoke \
  --request data/request.json
```

모델은 115,328 parameters. 훈련 12 records / 36 questions, 검증 3 records / 9 questions. 가장 긴 훈련 packed record는 byte tokenizer 기준 298 tokens였습니다.

| 항목 | 결과 |
| --- | --- |
| 초기 훈련 loss | 5.013699 |
| 30 epoch 최종 훈련 loss | 0.000138 |
| 최종 훈련 정확도 | 36/36 |
| 최종 검증 loss | 4.520702 |
| 최종 검증 정확도 | 4/9 |
| 타입별 검증 정확도 | choice 0/3, score 2/3, noul 2/3 |

저장된 모델을 복원해 훈련 예제의 질문 3개에 한 번에 답했습니다.

```json
{"route":"배송 문의","urgency":0,"refund":false}
```

실제 확률을 포함한 출력은 [smoke-inference.json](smoke-inference.json), 전체 학습 곡선은 [smoke-training.json](smoke-training.json), 타입별 검증 결과는 [smoke-evaluation.json](smoke-evaluation.json)에 보관했습니다. 기존 단일 choice 편의 CLI도 실행했습니다.

작은 예제에 과적합한 결과입니다. 구조와 학습/저장/추론 기능을 검증했지만, 일반적인 질문 응답 성능이나 CLEF 동등성을 입증하지는 않습니다. 모든 정답을 맞힌 훈련 요청과 별도의 검증 데이터 결과를 구분해야 합니다.

## 원래 MiniMind LM 변환

수정하지 않은 형제 프로젝트 `minimind/model/model_minimind.py`의 실제 `MiniMindForCausalLM`으로 무작위 초기화된 작은 native LM(hidden 32, 1 layer, vocab 6400) fixture를 만들었습니다. 이 LM의 `.pth`, config와 원래 로컬 HF tokenizer로 head-only 학습을 1 epoch 실행했습니다.

```bash
python train.py \
  --lm-checkpoint /tmp/minimind-j-v2-native/lm.pth \
  --lm-config /tmp/minimind-j-v2-native/config.json \
  --tokenizer ../minimind/model --freeze-backbone --epochs 1 \
  --output /tmp/minimind-j-v2-native/decision
```

저장된 체크포인트를 복원해 backbone의 **모든 tensor가 원본과 bitwise 일치**하고, 기존 LM head weight는 실제로 학습되어 달라졌음을 확인했습니다. tied 입력/출력 embedding은 분리되어 저장·복원되었습니다. 사전학습된 대형 모델의 의미 판단 성능 검증은 아닙니다.

원래 MiniMind 파일의 SHA256 `548f1203…c3786f`가 변경되지 않았음을 확인했습니다. 전체 해시는 `NOTICE`에 기록돼 있습니다. 형제 프로젝트와 GPU 자원은 수정하지 않았습니다.

## 독립 실행

형제 프로젝트 없이 `/tmp/minimind-j-v2-standalone`로 소스만 복사한 뒤 같은 pytest를 실행하고, Python compileall을 실행했습니다. 이전 후보별 scalar head 방식의 과거 검증은 [verification-v1.md](verification-v1.md)에 보관했습니다. 임시 checkpoint는 `/tmp`에 있으며 재현은 README 명령을 사용하세요.
