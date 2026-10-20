> 과거 별도 tiny 인코더 구현의 기록입니다. 현재 SigLIP-only 결과는 verification.md를 참고하세요.

# MiniMind-A 검증 기록

2026-10-20, macOS arm64, Python 3.11.11, CPU, 1 thread.
검증 환경: PyTorch 2.14.1 / Transformers 5.20.2 / NumPy 2.4.6 / Pillow 13.0.0.

디스크 여유 공간이 작아 PyTorch 중복 설치를 중단했습니다. 임시 venv
`/tmp/minimind-a-venv`가 기존 CPU 환경의 설치된 패키지를 재사용하고 Pillow만 추가했습니다.
프로젝트 코드는 sibling 프로젝트를 import하지 않습니다.
Transformers의 torchvision 미설치 시 PIL backend fallback 안내와 macOS distributed 안내가
출력되었으며 테스트·학습·추론에는 영향을 주지 않았습니다.

## 자동 테스트

프로젝트 루트에서:

```bash
python -m pytest -q
python -m compileall -q model dataset scripts train_vla.py eval_vla.py
```

17 passed. 검증 대상:

- flow의 t=0/1 끝점과 속도 정답
- scalar loss, 이미지·projector·언어 attention·행동 출력층에 대한 유한한 gradient
- 행동 패딩의 정답을 바꿔도 유효 loss 불변, 모든 행동이 패딩인 샘플 거부
- Euler 청크 생성에서 context 1회 계산, 출력 shape·finite·재현성·train mode 복원
- 이미지·지시·상태를 각각 변경하면 생성 행동이 바뀜
- 지시문 패딩의 토큰 ID를 바꿔도 생성 행동 불변
- VLM 동결 시 eval mode와 bitwise 가중치 보존
- 체크포인트 저장·복원 및 native VLM state_dict 초기화, 불일치 파일 거부
- 잘못된 이미지 prefix·행동 shape·추론 steps 거부
- 실제 PNG 처리, UTF-8 토큰, 배치 가변 길이·마스킹
- 잘못된 상태·행동 차원, 빈 행동·지시문, 비유한 상태 거부
- 로컬 tiny SigLIP 모델의 실제 from_pretrained·processor 경로에서 원본 MiniMind-V와
  최종 hidden states 일치; native trainer처럼 vision key를 제외한 가중치 로드
- 비전 폴더 누락 시 명확한 오류

SigLIP 테스트는 Hugging Face 구현의 작은 무작위 모델을 테스트 폴더에 저장해 실행합니다.
네트워크 다운로드·대형 사전학습 가중치·실제 로봇 데이터는 사용하지 않았습니다.

## 실제 학습·저장·추론

```bash
python scripts/make_toy_data.py --count 80
python train_vla.py --val-data data/toy/val.jsonl --epochs 30 --batch-size 16 --output out/toy
python eval_vla.py --output out/toy/prediction.json
```

- 기본 모델: 308,178 parameters.
- 64 train / 16 validation 독립 합성 장면. 30 epochs, 120 optimizer steps.
- 고정 noise/time validation flow MSE: 1.3823649883 → 0.6353214979.
- 첫 validation 장면 생성 행동 MSE(seed=42, Euler 10 steps): 0.1430279613.
- 전체 16개 validation 장면의 단일 noise sample 생성 MSE: 0.2675263286.
- 같은 전체 장면에서 항상 0을 반환하는 baseline MSE: 0.1251270324.
- 무작위 초기 모델 생성 MSE: 1.2368566990.

학습 모델은 무작위 초기 모델보다 나아졌지만 **0 baseline보다 나쁜 결과**입니다.
Flow loss 개선이 좋은 행동 정책이나 시각 grounding 성능을 증명하지 않습니다.
modalities 테스트는 입력 의존성만 확인합니다. 실제 로봇 성능은 측정하지 않았습니다.

설정·가중치·학습 기록·예측은 `out/toy/`, 재생성 가능한 데이터는 `data/toy/`에 있습니다.
`action_metrics.json`은 전체 validation의 위 행동 지표입니다.

## 독립 실행과 원본 보존

`minimind-a`를 `/tmp/minimind-a-isolated`로 복사하고 기존 data/out/cache는 제외했습니다.
그 디렉터리에서 17개 테스트를 통과하고, 새 8장 데이터를 생성해 1 epoch 학습,
체크포인트 저장·복원·행동 추론까지 실행했습니다. sibling 코드 없이 동작합니다.
포맷 정리 후 원본 디렉터리의 17개 테스트도 다시 통과했습니다.

복사한 원본 모델·tokenizer·LICENSE의 SHA-256은 원본 MiniMind-V 파일과 일치합니다.
`provenance.json`에 기록했습니다. 새 구현만 포맷했고 복사한 모델 파일은 바꾸지 않았습니다.
새 파일의 F 계열 lint 검사와 formatter 검사를 통과했습니다.
유료 GPU/클라우드 작업은 실행하지 않았습니다.

## 구조도

`architecture.mmd`와 README의 Mermaid 소스를 제공합니다. 외부 SVG 렌더링 서비스는
타임아웃되어 SVG를 포함하지 않았습니다. GitHub 등 Mermaid 지원 뷰어에서 볼 수 있습니다.
