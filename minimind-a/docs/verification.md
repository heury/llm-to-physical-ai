# 앞쪽 절반 레이어 구현 검증

2026-10-20: `python -m pytest -q` — 37개 통과.

- 원본 2/5/8 레이어에서 1/2/4개만 구성·실행되고 gradient가 흐르는지 검사.
- 원본 MiniMind-V 앞쪽 레이어와 가중치 및 hidden states 일치.
- 새 checkpoint format 2 저장·복원 후 동일 노이즈의 행동 출력 일치.
- 무작위 언어부 학습 및 원본 가중치 로드 후 동결 학습 각각 1 epoch, 저장·복원 통과.
- 실제 학습된 `sft_vlm_768.pth`와 SigLIP으로 CPU 검사:
  원본 8개 → 사용 4개, 총 132,594,439 parameters (`state/action_dim=7`).
  실제 이미지 입력, finite loss/backward, 동결 VLM gradient 없음,
  행동 expert gradient 유한함, Euler 2 steps 출력 `[1,16,7]` 모두 유한함.

실제 로봇 데이터 재학습이나 정책 성능 비교는 이 변경에서 수행하지 않았습니다.

## 이전 SigLIP-only 수정 검증

> 아래 기록은 앞쪽 절반 레이어로 전환하기 전의 전체 레이어 구현 결과입니다.
> 현재 구현은 원본 8개 중 앞쪽 4개를 사용하며 VLA 체크포인트 format 2로 새 학습이 필요합니다.
> 아래 오차·파라미터 수는 현재 모델의 성능으로 해석하지 마세요.

2026-10-20, macOS arm64, Python 3.11.11, CPU 1 thread.
PyTorch 2.14.1 / Transformers 5.20.2 / NumPy 2.4.6 / Pillow 13.0.0.

## 변경 사항

- 별도 `TinyVisionEncoder`와 byte tokenizer를 제거했습니다.
- 기본 입력은 MiniMind-V의 SigLIP processor와 6400-token tokenizer를 사용합니다.
- 기본 비전 폭/토큰 수는 768/64이며 SigLIP은 항상 동결합니다.
- `model/siglip2-base-p32-256-ve`는 기존 MiniMind-V 학습 결과의
  `vision_encoder` 디렉터리를 가리킵니다. 실제 pretrained 파일을 그대로 읽습니다.
- VLA 체크포인트에서 고정 SigLIP은 제외하고 복원 시 외부 비전 폴더에서 읽습니다.
- 기존 tiny 결과는 `out/legacy_tiny`에 보존했습니다. 그 결과는 현 구현의 성능이 아닙니다.

## 테스트

```bash
python -m pytest -q
python -m compileall -q model dataset scripts train_vla.py eval_vla.py
```

21개 테스트 통과. 기존 flow 끝점, 입력 의존성, 패딩 불변성, 문맥 1회 계산,
원본 VLM 문맥 일치, 동결, 저장·복원, native VLM 초기화 검사를 SigLIP 경로로 변경했습니다.
기본 SigLIP/tokenizer 설정, 제거된 인코더 거부, 체크포인트 비전 중복 저장 방지,
행동 가중치가 누락된 파일 거부도 검사합니다.
테스트 fixture는 작은 Hugging Face SigLIP 구조를 로컬에 무작위 초기화한 것이며,
별도의 인코더 구현이나 다운로드를 사용하지 않습니다.

## 실제 사전학습 SigLIP CPU 학습·추론

```bash
python scripts/make_toy_data.py --output data/siglip_smoke --count 8
python train_vla.py --data data/siglip_smoke/train.jsonl \
  --val-data data/siglip_smoke/val.jsonl --epochs 1 --batch-size 2 \
  --output out/siglip_smoke
python eval_vla.py --checkpoint out/siglip_smoke/vla.pt \
  --data data/siglip_smoke/val.jsonl --output out/siglip_smoke/prediction.json
```

기존 MiniMind-V 학습 산출물의 실제 SigLIP 가중치를 로드했습니다.
작은 언어 Transformer는 무작위 초기화이며 전체 pretrained VLM을 사용하는 아래 검사와 구분합니다.

- 전체 모델 95,262,530 parameters; SigLIP은 고정.
- 6 train / 2 validation 장면, 1 epoch, 3 optimizer steps.
- train flow loss 1.1238113.
- 고정 noise/time validation flow loss 1.3361074 → 1.2124377.
- 저장 후 첫 validation 장면에서 `[1,8,2]` 행동 생성 성공; MSE 1.1807538.
- SigLIP을 제외한 `out/siglip_smoke/vla.pt`는 약 2.7 MiB.

실행 경로 검증용의 짧은 학습입니다. 행동 성능·시각 grounding·실제 로봇 성공률을 검증하지 않았습니다.

## 학습된 MiniMind-V 전체 재사용 검사

`configs/minimind_v_768.json`으로 실제 SigLIP을 구성한 뒤 기존의
`../outputs/minimind-v/sharegpt4v-20261020/sft_vlm_768.pth`를 로드했습니다.

- 768 hidden / 8 language layers 원본 가중치 로드 성공.
- processor 이미지 shape `[1,3,256,256]`.
- 현재 상태 `[1,7]`, 생성 청크 shape `[1,16,7]`.
- 행동 loss 1.3374718, backward 성공.
- 동결 VLM gradient 없음, 행동 expert gradient 유한함.
- Euler 2 steps 생성 행동이 모두 유한함.

로봇 정책 학습을 수행한 검사는 아닙니다. 결과는
`out/siglip_smoke/native_vlm_check.json`에 저장했습니다.

## 환경과 원본 보존

임시 `/tmp/minimind-a-venv`에서 기존 설치된 CPU 패키지를 재사용했습니다.
SigLIP processor의 PIL fallback 및 macOS distributed 안내가 출력되지만 실행은 정상입니다.
수정한 코드의 F 계열 lint, formatter, compileall을 확인했습니다.
원본 복사 파일과 tokenizer·LICENSE는 `provenance.json`의 SHA-256과 일치합니다.
기존 프로젝트·사전학습 가중치를 수정하지 않았으며 유료 GPU를 사용하지 않았습니다.

이전 구현의 검증 기록은 `verification-tiny-legacy.md`에 별도로 보존했습니다.
