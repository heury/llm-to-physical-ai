# 검증 기록

검증일: 2026-10-25 (Asia/Seoul)

## 실행 환경과 결과

- Python: 기존 `../minimind/.venv/bin/python` 사용.
- PyTorch: `2.14.1+cu130`, Transformers: `4.57.6`.
- 테스트 장치: CPU. 외부 모델·데이터 다운로드 없음.
- 명령: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../minimind/.venv/bin/python -m unittest discover -s tests -v`
- 결과: 5개 테스트 통과, 최초 실행 26.244초.

## 검증 내용

1. `test_padding_and_constrained_logits`: 무작위 32차원/1층 Dense 및 MoE MiniMind에 실제 로컬 토크나이저 입력. 길이가 다른 두 질문의 단독/배치 로짓과 원본 전체 LM head에서 추출한 A–E 로짓이 허용 오차 내 일치.
2. `test_only_head_updates_and_loss_decreases`: 공유 가중치를 분리하고 12 optimizer step 실행. 손실 감소, 본체 전체 파라미터와 비선택지 head 행 불변, 본체 gradient 없음. 저장·재로드 후 공유 해제와 로짓 일치.
3. `test_validation_and_no_label_leakage`: answer만 바꿔도 프롬프트 불변. 잘못된 입력, 정답, 빈 데이터, 초과 길이, 다중 토큰 선택지 거부.
4. `test_calibration`: 온도 양수, fitting-set NLL 비증가, argmax 정확도 불변, 0/음수/NaN/무한대 온도 거부.
5. `test_cli_lifecycle`: 임시 native 가중치에서 학습 2 epoch → 저장 → 평가 → 보정 → 추론 실행. 확률 합 1, 기존 결과 덮어쓰기 거부, 다른 모델 해시의 보정 거부.

CLI 통합 테스트는 기능 연결을 검증하기 위해 같은 합성 샘플을 학습·보정·평가에 사용합니다. 이 수치는 독립 평가 성능으로 해석하지 않습니다. 실제 데이터 준비 명령은 calibration/eval을 분리합니다.

## 실행하지 않은 검증

- CommonsenseQA 다운로드 및 전체 학습/평가.
- 사전 학습된 MiniMind 가중치를 사용한 상식 판단 성능 평가.
- GPU 학습, 대규모 MoE 학습, 외부 tokenizer/가중치 조합 검증.

이 프로젝트에는 학습 완료된 Jev 가중치를 포함하지 않습니다. 추론에는 사용자가 준비한 MiniMind native 가중치 또는 `train`으로 만든 체크포인트가 필요합니다.
