# GSM8K GRPO 보상 비교 학습

기록일: 2026-10-25 (KST)

사용자 요청과 최종 결과를 기록했습니다. 중간 진행 알림과 도구 실행 로그는 생략했습니다.

## user — 보상 함수 작성

openai/gsm8k을 이용하여 grpo 학습을 진행할려고 해. 보상함수를 작성해봐

## user — 100개 샘플 테스트

[https://huggingface.co/datasets/caspar/MiniMind-Mixed-SFT-Training-v1.0](https://huggingface.co/datasets/caspar/MiniMind-Mixed-SFT-Training-v1.0)에서 gsm8k 데이터에서 100개를 뽑아서 보상 함수를 테스트해봐

## user — 검토 및 수정

테스트한 결과를 분석해서 개선할 부분이 있는지 검토해

검토한 결과를 바탕으로 수정해

## user — 학습 및 업로드

caspar/MiniMind-Mixed-SFT-Training-v1.0에서 gsm8k 데이터를 뽑아서 grpo 학습을 진행해. 학습은 작성한 reward 함수와 reward 모델 Skywork/Skywork-Reward-V2-Qwen3-1.7B을 사용한 두가지를 순차적으로 진행해. 학습 후 결과와 샘플 테디트를 보고하고, 모델은 huggingface에 업로드해


## agent — 최종 결과

**Dense와 MoE 모두 학습·평가를 완료하고, 네 모델을 Hugging Face에 공개했습니다.**

각 실험은 같은 SFT 체크포인트에서 독립적으로 시작해, GSM8K **6,851문제 전체를 1epoch·215회 업데이트**했습니다. 평가는 별도 테스트 **1,319문제**에 동일한 정답 채점 함수를 적용했습니다.

| 모델 | 학습 전 | 규칙 보상 GRPO | Skywork 보상 GRPO |
|---|---:|---:|---:|
| Dense 64M | 14개 · 1.06% | [18개 · 1.36%](https://huggingface.co/caspar/MiniMind-Dense-64M-GSM8K-GRPO-Rule-v1.0) | [12개 · 0.91%](https://huggingface.co/caspar/MiniMind-Dense-64M-GSM8K-GRPO-Skywork-v1.0) |
| MoE 198M-A64M | 16개 · 1.21% | [26개 · 1.97%](https://huggingface.co/caspar/MiniMind-MoE-198M-A64M-GSM8K-GRPO-Rule-v1.0) | **[32개 · 2.43%](https://huggingface.co/caspar/MiniMind-MoE-198M-A64M-GSM8K-GRPO-Skywork-v1.0)** |

고정된 테스트 첫 두 문제의 실제 최종 답은 다음과 같습니다. 모두 오답이며, 성공 사례만 선택하지 않았습니다.

| 샘플 | 정답 | Dense 규칙 | Dense Skywork | MoE 규칙 | MoE Skywork |
|---|---:|---:|---:|---:|---:|
| 오리알 판매 수익 | 18 | 6 | 36 | 8888 | 128 |
| 로브 제작에 필요한 천 | 3 | 8 | 36 | 8 | 8 |

**MoE + Skywork이 가장 높았지만, 절대 정확도는 여전히 낮습니다.** 특히 Dense + Skywork은 보상 모델 점수가 상승했는데도 정답률은 떨어졌습니다. 보상 점수 상승만으로 수학 능력이 개선됐다고 판단할 수 없습니다.

네 모델 모두 `v1.0` 태그와 전체 평가 응답을 포함하며, 재로드·가중치 해시·공개 다운로드를 검증했습니다. 작업용 GPU Pod는 모두 삭제했고 기존 볼륨은 보존했습니다.

[상세 보고서와 전체 샘플 응답](../outputs/gsm8k-grpo-20261025/report.md)에서 학습 설정, 보조 점수, 중간 오류 복구 내역을 확인할 수 있습니다.

## 관련 구현 및 산출물

- [규칙 보상 함수](../trainer/gsm8k_reward.py)
- [GSM8K 데이터 로더](../dataset/gsm8k_dataset.py)
- [보상 평가 스크립트](../scripts/eval_gsm8k_rewards.py)
- [100개 샘플 보상 테스트 산출물](../outputs/gsm8k-reward-sample-100/)
- [GRPO 비교 실험 실행 안내](../scripts/runpod_gsm8k_grpo/README.md)
- [최종 검증 기록](../outputs/gsm8k-grpo-20261025/final_audit.json)

## user — 대화 저장

이 대화를 prompts에 저장해
