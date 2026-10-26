# DPO 학습

기록일: 2026-10-25 (KST)

## user

`argilla/ultrafeedback-binarized-preferences-cleaned` 에서 340 토큰 미만만 필터링해서 3k 정도 데이터를 만들어서 dpo 학습을 진행해. caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0
이 모델을 이용해. 학습 후 학습 결과와 샘플 데이터를 보고하고 학습한 모델과 생성한 데이터셋은 huggingface에 업로드해. 작업이 끝나면 pod는 제거해

## agent

학습·평가·공개 업로드를 완료했고, **Pod도 삭제했습니다.**

- [학습 모델 v1.0](https://huggingface.co/caspar/MiniMind-Dense-64M-UltraFeedback-DPO-3K-v1.0)
- [필터링한 데이터셋 v1.0](https://huggingface.co/datasets/caspar/MiniMind-UltraFeedback-Short340-DPO-3K-v1.0)
- [W&B 학습 로그](https://wandb.ai/mind4u/MiniMind-DPO-Short/runs/4gurvz7a)

chosen·rejected 각각 **프롬프트와 특수 토큰을 포함해 340토큰 미만**인 쌍을 선별했습니다. 문장 절단 없이 총 3,000쌍을 학습 2,700 / 검증 150 / 테스트 150으로 나누고, 2에포크·170회 업데이트했습니다.

**테스트 150쌍 결과**

| 지표 | 학습 전 | 학습 후 |
|---|---:|---:|
| DPO 손실 ↓ | 0.6931 | **0.6457** |
| 문장 로그확률 기준 선호 판별률 | 46.67% | 46.67% |
| Chosen 응답 PPL ↓ | 5.6424 | 5.6708 |

DPO 손실은 감소했습니다. **명확한 응답 품질 향상은 확인하지 못했으며, 계산 오류와 반복 출력은 남아 있습니다.**

실제 학습 데이터 예시—chosen 43토큰, rejected 44토큰:

```text
Prompt: what does "waning moon" mean?
        a) lighted B) shrinking C) dark D) growing

Chosen: B) shrinking
Rejected: D) Growing.
```

생성 샘플에서 `What is 17 + 25?`의 최종 답은 학습 전 **25**, 학습 후 **5**로 모두 오답이었습니다. 정답은 42입니다. [전체 16개 학습 전후 샘플](https://huggingface.co/caspar/MiniMind-Dense-64M-UltraFeedback-DPO-3K-v1.0/blob/v1.0/samples.json)도 공개했습니다.

공개 로딩·가중치 해시 검증을 완료했고, 결과는 shared-volume에도 보존했습니다.
