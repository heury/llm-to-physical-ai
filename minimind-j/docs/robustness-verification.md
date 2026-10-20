# 질문 수 정규화 및 강건성 평가 검증

2026-10-20. `/tmp/minimind-j-venv/bin/python -m pytest -q`: **41 passed**.

- 서로 다른 질문 수 `[1, 3, 2]` 및 짧은 구간 `[2, 1]`으로 분할한 gradient가 전체 질문 평균 loss의 gradient와 일치.
- 질문 표현 변경에서 rubric 보존, choice 역순에서 soft gold 재배열 및 원본 불변 확인.
- 후보 실제 값으로 분포를 정렬하고, 동률에서도 실제 decoder의 선택 변경을 감지.
- 기존 모델 학습·저장·복원·질문 격리 테스트 모두 통과.

기존 `/tmp/minimind-j-v2-smoke`의 115k 예제 모델을 `data/validation.jsonl`
3 records / 9 questions에 평가했다. 모델을 새로 학습하지 않았다.

```bash
python scripts/evaluate_robustness.py \
  --checkpoint /tmp/minimind-j-v2-smoke --data data/validation.jsonl \
  --paraphrases scripts/example_paraphrases.json \
  --output docs/robustness-smoke.json
```

| 조건 | 정확도 | 답변 변경률 | choice 답변 변경률 | 평균 total variation |
| --- | --- | --- | --- | --- |
| 원본 | 4/9 | 0 | 0 | 0 |
| 질문 표현 변경 | 2/9 | 5/9 | 1/3 | 0.5281 |
| choice 후보 역순 | 6/9 | 3/9 | 3/3 | 0.3240 |

후보 역순에서 정확도가 상승했어도 choice 답변이 전부 달라지므로 순서에 강건하다고
해석할 수 없다. 작은 과적합 모델의 기능 검증이며 통계적인 성능 평가가 아니다.
기계 판독 결과는 `robustness-smoke.json`에 저장했다.

실제 64M 모델은 로컬 `out/typed-decisions-20261020/model/`에 설정과 tokenizer만
있고 `decision.pth`가 없다. 따라서 해당 모델의 강건성 평가는 실행하지 않았다.
공유 볼륨 checkpoint를 사용할 수 있는 환경에서 runpod_typed/README.md의 명령으로
평가해야 한다. 이번 작업에서 GPU 생성, 원격 학습, 기존 실험 결과 수정은 하지 않았다.
