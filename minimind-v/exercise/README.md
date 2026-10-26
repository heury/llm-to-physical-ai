# MiniMind-V 기능 실습

대상 소스: [model_vlm.py](../model/model_vlm.py). 직접 만든 빨간 사각형과 파란 원 이미지로 **실제 MiniMindVLM 메서드**를 호출합니다. 한국어 주석, 중간 텐서 출력, `assert` 검증을 함께 읽으며 진행하세요.

학습 가중치, 데이터셋, GPU, API 키가 필요 없습니다. 작은 SigLIP을 무작위 초기화한 뒤 임시 디렉토리에 저장하여 실제 로딩 경로도 실행합니다. 의존성 설치 후에는 오프라인 실행이 가능합니다. 기존 모델·학습 데이터는 수정하지 않으며 임시 모델은 종료 시 삭제합니다.

## 실행

`run_exercises.py`에는 `# %% [번호] 제목` 셀 구분자가 있습니다. VS Code의 Python/Jupyter 확장에서 해당 파일을 열고, 의존성이 설치된 커널을 선택하세요. **[1]~[4] 준비 셀을 순서대로 실행**한 뒤 **[5]~[11] 실습 셀**은 원하는 순서로 실행할 수 있습니다. 각 실습 셀은 함수를 정의하고 즉시 독립적인 초기 모델로 테스트합니다. `[12]`는 명령행 실행용이며 Interactive Window에서는 자동 실행되지 않습니다. 커널 실행에 필요하면 환경에 `ipykernel`을 설치하세요.

저장소 루트에서 다음과 같이 실행합니다. 기존 프로젝트 환경에 의존성이 있다면 환경 생성과 설치를 생략하세요.

```bash
cd minimind-v
python3 -m venv exercise/.venv
source exercise/.venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r exercise/requirements.txt
HF_HUB_OFFLINE=1 python exercise/run_exercises.py --case all
```

한 기능씩 실행하려면:

```bash
python exercise/run_exercises.py --case 3
python exercise/run_exercises.py --case 5
```

실습마다 초기화하므로 순서에 의존하지 않습니다. 정상 종료 시 각 단계에 `PASS`, 마지막에 `완료: 7개 실습 통과`가 출력됩니다. 검증 실패 시 예외와 종료 코드 1이 발생합니다. `assert`를 비활성화하는 `python -O`는 사용하지 마세요.

## 실습 구성

| 번호 | 대상 기능 | 확인할 내용 |
| --- | --- | --- |
| 1 | `VLMConfig`, `get_vision_model`, `image2tensor` | 로컬 모델 로딩, 비전 인코더 동결, 없는 경로 처리, RGBA→RGB 전처리 |
| 2 | `get_image_embeddings`, `MMVisionProjector` | 패치 특징 추출, gradient 차단, 언어 임베딩 차원으로 투영 |
| 3 | `count_vision_proj` | 단일·복수 이미지의 marker 구간 교체, 텍스트 위치 보존, 이미지 없을 때 처리 |
| 4 | `forward` | 텍스트만/단일·복수 이미지 입력, logits shape, `logits_to_keep` 선택 |
| 5 | `forward(labels=...)` | 답변만 대상으로 한 shifted cross entropy, 역전파, projector만 갱신 |
| 6 | `past_key_values`, `use_cache` | 전체 재계산과 캐시 결과 비교, 다음 토큰에서 비전 인코더 재실행 생략 |
| 7 | `generate` | greedy·sampling, `num_return_sequences=2`, 캐시 유무 결과 비교 |

## 샘플을 읽는 법

실습 모델은 언어 hidden size 32, 2개 층, vocabulary 64, 비전 hidden size 16입니다. 이미지 32×32를 16×16 패치로 나누므로 시각 토큰은 4개입니다. 실제 기본 모델의 256×256 이미지·64개 시각 토큰보다 작게 만들었습니다.

```text
가상 토큰: [BOS, IMAGE, IMAGE, IMAGE, IMAGE, 질문, 답변, EOS]
토큰 ID:  [  1,    12,    12,    12,    12,   20,   21,   2]
labels:   [-100, -100, -100, -100, -100, -100,   21,   2]

이미지 [1,3,32,32]
 → SigLIP 특징 [1,4,16]
 → projector [1,4,32]
 → IMAGE 위치의 언어 임베딩 교체
 → 언어 모델 logits [1,8,64]
```

`20`, `21`은 문장을 토크나이즈한 값이 아닌 설명용 ID입니다. 무작위 모델의 생성 ID는 자연어 답변이나 이미지 인식 정확도를 의미하지 않습니다. 실습 5는 실제 optimizer step을 수행하지만, 한 번의 갱신으로 색상이나 사물 이름을 학습했다고 해석하면 안 됩니다.

## 현재 구현에서 주의할 입력 조건

- `MMVisionProjector`의 `source_tokens`, `target_tokens` 인자는 현재 토큰 수를 변경하지 않습니다. 인코더 출력 패치 수와 `image_token_len`, 연속 IMAGE marker 수를 일치시켜야 합니다.
- 복수 이미지는 `{'pixel_values': tensor[B,N,C,H,W]}`로 전달합니다. 두 이미지의 marker 구간 사이에 텍스트 토큰을 두어야 각각 다른 이미지로 처리됩니다. 붙어 있는 marker는 한 구간으로 해석됩니다.
- 단일 이미지는 processor가 반환하는 딕셔너리 `[B,C,H,W]` 경로를 사용합니다. 현재 원시 텐서 분기는 내부에서 텐서를 `**image_inputs`로 넘기는 문제가 있어 이 실습의 지원 입력에 포함하지 않습니다.
- 경로가 없으면 비전 인코더가 `None`이 됩니다. 텍스트만의 forward는 가능하지만 이미지 입력에는 실제 인코더가 필요합니다.
- 실습 5는 projector 학습을 보여 주기 위해 언어 모델도 고정합니다. 실제 SFT trainer의 층 선택 정책과는 별개의 최소 예제입니다. loss는 다음 토큰 예측을 위해 내부에서 한 칸 shift합니다.
- 전체 labels와 `logits_to_keep` 축소를 함께 사용하지 않습니다. loss 계산에서는 전체 logits를 사용하세요.

## 수강생 추가 과제

1. 실습 2의 `target_tokens`를 바꾸고 토큰 개수가 왜 그대로인지 projector 코드를 설명하세요.
2. 실습 3의 시각 특징을 3과 4로 바꾸고 텍스트 위치 보존을 다시 확인하세요.
3. 실습 5에서 labels를 모두 `-100`으로 만들었을 때 loss를 관찰하고, 학습 데이터에 답변 토큰이 필요한 이유를 설명하세요.
4. 실습 7에서 seed와 temperature를 바꾸어 생성 ID 변화를 비교하세요. greedy 두 출력이 같은 이유도 설명하세요.

이 실습은 구조·수치·gradient 경로를 확인합니다. 실제 체크포인트의 이미지 설명 품질, tokenizer/chat template, 데이터 로더, MoE, GPU/DDP 학습은 별도 검증 대상입니다.

## 실행 검증 기록

2026-10-25, Python 3.13.13 / PyTorch 2.6.0+cpu / Transformers 4.57.6 / Pillow 13.0.0 환경에서 `HF_HUB_OFFLINE=1`로 전체 7개 실습을 실행하여 통과했습니다. 실습 5의 초기 loss는 4.0954였으며 projector의 6개 매개변수 텐서만 변경되었습니다. 이는 무작위 모델의 기능 확인 결과이며 학습된 VLM의 성능 점수가 아닙니다. 의존성 버전에 따라 세부 수치와 생성 ID는 달라질 수 있습니다.
