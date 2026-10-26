# 학습

## user
pretrain 모델을 sft fune tuning해. 
HuggingFaceH4/ultrachat_200k
HuggingFaceTB/smoltalk
openai/gsm8k
이 세개 데이터를 이용해.

## agent
**RTX 5090 Pod 한 대를 생성했고, 현재 세 데이터셋을 전처리 중입니다.** 본 SFT 업데이트는 준비와 기준 평가가 끝나면 자동으로 시작합니다.

| 모델 | Pod |
|---|---|
| Dense | `kffunrj2g9vm9a` |

- 데이터: UltraChat `train_sft` + SmolTalk `all/train` + GSM8K `main/train`
- 설정: **2 epoch, 최대 길이 340**, 기존 토크나이저 유지
- 답변과 **EOS를 보존**하고, 테스트 중복·대화 중복 제거
- 요금: 두 대 합계 **$2.38/시간**

별도 감시 서비스가 W&B 기록, 학습 전후 영어·GSM8K 평가, 완료 검증과 **Pod 자동 종료**까지 수행하도록 실행 중입니다.

결과 저장 경로: `/shared-volume/minimind-mixed-sft-20261024/`

[현재 상태·설정·접속 정보](/home/mind4u/dev/minimind/outputs/minimind/mixed-sft-20261024/launch_summary.md)


# 결과 보고

## user
학습 결과 보고해

## agent 
**10월 25일 07:36 KST 기준, Dense는 학습·평가가 완료됐고 MoE는 학습 중입니다.** UltraChat·SmolTalk·GSM8K 혼합 SFT 결과입니다.

| 항목 | Dense | MoE |
|---|---:|---:|
| 진행 상태 | 2 epoch 완료 | 2번째 epoch, 전체 약 74.2% |
| 업데이트 | 84,320 / 84,320 | 62,570 / 84,320 |
| 학습 시간 | 약 5시간 19분 | 약 2시간 17분 남음 |
| Pod | 삭제 확인 | 실행 중 |

MoE 학습은 현재 속도라면 **09:55 전후** 완료 예상이며, 이후 평가·저장 검증 시간이 추가됩니다.

**Dense는 검증 PPL이 개선됐지만, 수학 정확도는 여전히 낮습니다.**

| 평가 | SFT 전 | SFT 후 |
|---|---:|---:|
| 전체 검증 PPL ↓ | 11.71 | **3.79** |
| UltraChat PPL ↓ | 11.08 | **4.90** |
| SmolTalk PPL ↓ | 11.49 | **3.54** |
| GSM8K PPL ↓ | 13.36 | **2.69** |
| GSM8K 엄격 정답률 | 0 / 1,319 | **15 / 1,319 (1.14%)** |
| 마지막 숫자 기준 정답률 | 5 / 1,319 | **23 / 1,319 (1.74%)** |

실제 출력에서도 문제가 남아 있습니다.

- `7 × 8`을 **48**로 계산했습니다.
- 영어 학습 팁을 정확히 3개 요청했지만 **5개와 반복 표현**을 출력했습니다.
- 새는 깃털이 있다는 전제를 줬는데도 `"No, a robin does not have feathers."`라고 답했습니다.
- 기초 계산은 자동 채점상 1/6이지만, 맞힌 답도 풀이가 잘못되어 신뢰하기 어렵습니다.

Dense 가중치는 다음 위치에 저장했고, 재로딩·체크포인트 검증을 통과했습니다.

```text
/shared-volume/minimind-mixed-sft-20261024/dense/out/full_sft_768.pth
```

[Dense W&B](https://wandb.ai/mind4u/MiniMind-Mixed-SFT/runs/pv51fm2v) · [MoE W&B](https://wandb.ai/mind4u/MiniMind-Mixed-SFT/runs/n9ge1i1i)

MoE 최종 성능 비교는 아직 나오지 않았습니다. 완료 후 평가·저장 검증과 Pod 종료는 자동으로 진행됩니다.