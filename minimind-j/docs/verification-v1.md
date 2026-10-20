# v1 CPU 검증 기록 (과거 구조)

현재 v2 결과는 [verification.md](verification.md)를 참고하세요. 아래는 교체 전의 후보별 scalar head 구조를 검증한 기록입니다.

2026-10-20, macOS arm64 / Python 3.11.11 / PyTorch 2.14.1 / Transformers 5.20.2 / pytest 9.1.1.

## 자동 테스트

```bash
python -m pytest -q
```

**16 passed**. 형제 프로젝트 없이 `/tmp/minimind-j-standalone-check`에 소스만 복사한 환경에서도 **16 passed**, Python compileall도 통과했습니다. 다음 동작을 실제 tensor와 작은 MiniMind로 검증했습니다.

- 가변 후보 수의 padding 확률 0, 확률 합 1, 유한 loss와 backbone/head gradient.
- 배치의 다른 샘플 때문에 추가된 token padding이 기존 샘플 점수를 바꾸지 않음.
- 후보 순서를 바꾸면 점수 순서만 바뀜.
- 작은 데이터에 대한 loss 감소와 정답 학습.
- 저장·재로드 시 logit 일치, byte/HF tokenizer 모두 복원.
- native LM state_dict 변환, 누락된 backbone key 거부.
- backbone 동결 후 실제 optimizer step에도 backbone tensor가 그대로 유지됨.
- SDPA와 수동 attention 경로의 점수 일치.
- 잘못된 label/빈 후보/중복 후보/후보 누락/최대 길이 초과/label 혼합 거부.

## CLI 학습과 추론

```bash
python train.py --epochs 30 --validation data/validation.jsonl \
  --output /tmp/minimind-j-smoke
python decide.py --checkpoint /tmp/minimind-j-smoke \
  --input '배송이 언제 오나요?' \
  --choices '배송 문의' '환불 요청' '계정 문의'
```

119,552 parameters, CPU 학습. 초기 훈련 loss **0.996994**, 최종 **0.001026**, 훈련 정답 **12/12**. 저장된 체크포인트를 재로드한 CLI는 `배송 문의`를 선택했고 확률은 약 **0.999912**였습니다. 이 문장은 훈련 예제입니다.

별도 검증 예제 3개에서는 최종 **1/3**, loss **3.383630**입니다. 중간 epoch에서 더 좋은 값이 나왔지만 마지막 모델은 과적합했습니다. 예제 규모가 너무 작기 때문에 일반적인 고객 문의 분류 성능이나 CLEF와의 동등성을 입증하는 결과가 아닙니다. epoch별 수치는 [smoke-training.json](smoke-training-v1.json)에 보관했습니다. 이 구현은 마지막 epoch 모델을 저장하며 best checkpoint 선택은 생략했습니다.

## 실제 native MiniMind 포맷 변환 경로

원래 형제 프로젝트의 `MiniMindForCausalLM`으로 작은 **무작위 초기화** dense LM(hidden 32, 1 layer, vocab 6400)의 native state_dict와 config를 생성했습니다. 실제 로컬 MiniMind HF tokenizer를 사용해 다음 경로를 실행했습니다.

```bash
python train.py \
  --lm-checkpoint /tmp/minimind-j-native-test/lm.pth \
  --lm-config /tmp/minimind-j-native-test/config.json \
  --tokenizer ../minimind/model \
  --freeze-backbone --epochs 1 \
  --output /tmp/minimind-j-native-test/decision
```

strict 로딩, head-only 학습, tokenizer 저장, 독립 checkpoint 추론까지 실행했습니다. 변환 후 backbone의 모든 tensor가 원본 LM과 bitwise 일치하는지 비교했습니다. 이는 파일 포맷 호환성 검증이며, 사전학습된 대형 체크포인트의 의미 판단 성능 검증은 아닙니다.

임시 가상환경과 체크포인트는 `/tmp`에 생성했습니다. 재현은 README의 설치 및 학습 명령을 사용하세요. 기존 MiniMind 소스와 원래 가중치는 수정하지 않았습니다.
