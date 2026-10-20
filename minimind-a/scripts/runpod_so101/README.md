# SO101 Runpod 학습

`caspar/so101_pickplace_sim` 리비전 `9b1c5189e3858e9254fc5e48259a035d22da497a`.
43 episodes / 7,396 frames / 20 Hz. 상단 카메라 한 개, 상태·행동 6차원.
원본 지시문: `Pick up the red cube and put it in the blue box.`

`run.py`는 이번 전용 B300 MIG Pod에서 실행하는 재현 스크립트입니다.
`ROOT`, `BASE`, 데이터 리비전은 파일 상단에 명시했습니다.

- `data_utils.py`: 고정 에피소드 split, 에피소드 경계를 넘지 않는 chunk, 학습 통계 정규화.
- `run.py`: 타임스탬프 기반 영상 정렬, native VLM 문맥 캐시, expert 학습·검증·저장.
- `watch.py`: 완료 및 전체 shared-volume 파일 해시 검증 후 지정한 Pod만 삭제.

기존 MiniMind-V의 학습된 768×8 VLM에서 앞쪽 4개 언어 레이어와 원본 SigLIP을 동결합니다.
4개 레이어를 거친 VLM hidden states를 한 번 계산해 캐시하며, 학습하는 context/state 투영과 행동 expert는
캐시에서 매번 새로 계산합니다. 따라서 현재 동결 설정에서 원래 forward와 같은 계산입니다.
캐시 파일 이름에 사용 레이어 수를 포함해 이전 전체 레이어 문맥과 구분합니다.
기존 8-layer VLA 체크포인트는 지원하지 않으며, 새 학습에는 별도 `ROOT`를 지정하세요.
이전 산출물의 평가 수치는 새 4-layer 모델의 결과가 아닙니다.

5개 에피소드만 검증에 사용하고 나머지 38개를 전부 학습합니다.
정규화 통계는 학습 프레임으로만 계산합니다. 각 관측에서 향후 16개 절대 joint target을 예측하며,
에피소드 끝 이후 시점은 마스킹합니다. validation은 독립 에피소드의 offline MSE입니다.
시뮬레이터 rollout/성공률은 측정하지 않습니다.

학습: 100 epochs, batch128, AdamW 1e-4, 100 steps warmup 이후 cosine→1e-5,
BF16 autocast, grad clip1. 원본 시간 convention은 noise(t=0)→data(t=1)입니다.

산출물은 `/shared-volume/minimind-a-so101-20261020/`에 저장합니다.

- `model/best_vla.pt`: 고정 noise/time validation flow MSE가 가장 낮은 시점.
- `model/final_vla.pt`: 100 epoch 최종 가중치. frozen SigLIP만 제외하고 전체 VLA를 포함.
- `model/training_state.pt`: 최종 optimizer 상태·epoch·update 횟수.
- `normalization.json`: raw state/action ↔ 정규화 값 변환 통계.
- `dataset_manifest.json`: train/validation 에피소드와 원본 파일 SHA.
- `evaluation_before.json`, `evaluation_after.json`, `evaluation_best.json`.
- `validation_predictions_{before,final,best}.npz`: raw 단위 예측·정답·유효 시점 마스크.
- `examples/`, `example_prediction.json`: 실제 프레임 기반 복원 추론 예시.
- `SHA256.json`, `SUCCESS`: 검증한 저장 파일과 완료 표시.

새 관측의 상태를 `normalization.json`의 state mean/std로 정규화해
`VLAProcessor` → `collate_batch` → `MiniMindVLA.sample_actions`로 입력합니다.
출력은 action std를 곱하고 mean을 더해 raw joint target으로 복원합니다.
복원에는 같은 shared-volume의 기존 `minimind-v-sharegpt4v-20261020/vision_encoder`가 필요합니다.
다른 머신으로 옮기면 `MiniMindVLA.load_checkpoint(..., vision_model_path=...)`를 지정하세요.
