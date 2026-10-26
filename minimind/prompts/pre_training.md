# 모델 학습

## user

fineweb-edu 10bt에서 점수가 3.5이상인 데이터를 shared-volume에 저장해 두었으니 이 데이터로 pretrain 모델 학습해. 최대 입력길이는 340, epoch은 2로 하고 나머지 옵션은 gpu 사양에 맞춰.

# 결과 보고

# user
진행 상황 보고해

# agent
**Dense·MoE 모두 사전학습을 완료했고, Pod 두 대도 종료됐습니다.** 10월 24일 22:09 KST에 재확인했습니다.

| 항목 | Dense | MoE |
|---|---:|---:|
| 학습 | 2 epoch 완료 | 2 epoch 완료 |
| 업데이트 | 34,432회 | 34,432회 |
| 최종 검증 loss | 2.6561 | **2.5778** |
| 최종 검증 PPL | 14.24 | **13.17** |
| 학습 소요 시간 | 약 4시간 22분 | 약 5시간 46분 |
| 완료 시각 | 15:23 KST | 16:50 KST |

두 모델 모두 저장 가중치 재로드·유한성·체크섬 검증을 통과했으며, W&B 서버도 `finished` 상태입니다. **수학 성능 개선 여부는 아직 별도 평가하지 않았습니다.**

모델은 `/shared-volume/minimind-full-pretrain-20261024/` 아래에 보존되어 있습니다.

- Dense: `dense/out/pretrain_768.pth`
- MoE: `moe/out/pretrain_768_moe.pth`

[결과 보고서](/home/mind4u/dev/minimind/outputs/minimind/full-pretrain-20261024/report.md) · W&B [Dense](https://wandb.ai/mind4u/MiniMind-FineWeb-Full-Pretrain/runs/shk6um5l) / [MoE](https://wandb.ai/mind4u/MiniMind-FineWeb-Full-Pretrain/runs/yelhyppc)