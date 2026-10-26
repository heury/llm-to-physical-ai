# MiniMind-J2 — MiniMind로 만드는 Jev

`/home/mind4u/dev/build-your-own-jev/scripts/`의 Jev 실험을 MiniMind로 구현한 독립 프로젝트입니다. 질문과 A–E 선택지를 한 번 인코딩하고, 마지막 입력 위치의 hidden state에 기존 LM head의 A–E 토큰 행을 적용해 다섯 선택지의 점수와 확률을 얻습니다. 자유 텍스트 생성이나 별도 분류 MLP를 사용하지 않습니다.

학습 시 입력 임베딩과 LM head의 가중치 공유를 해제한 다음 본체 전체를 고정하고 LM head만 업데이트합니다. 실제 손실에 참여하는 것은 A–E 행뿐입니다. 판단 질문을 packing하는 `minimind-j`와 달리 한 샘플은 하나의 5지선다 질문입니다. Dense와 MoE 모두 같은 경로를 사용합니다.

## 파일

- `jev.py`: 데이터 검증, 학습, 추론, 평가, temperature 보정, 데이터 준비 CLI.
- `model/model_minimind.py`: 형제 `minimind/model/model_minimind.py`에서 가져온 수정하지 않은 백본.
- `model/tokenizer*`, `model/vocab.json`, `model/merges.txt`: 같은 MiniMind 토크나이저의 로컬 사본.
- `configs/minimind_768.json`: 현재 기본 768차원/8층 Dense 모델 설정 예시.
- `data/input.json`: 원본의 bat 질문과 추가 추론 예제. 학습 데이터가 아닙니다.
- `tests/test_jev.py`: 작은 실제 MiniMind를 사용하는 단위·CLI 통합 테스트.
- `VALIDATION.md`: 검증 범위와 결과.

## 환경

아래 명령은 모두 이 프로젝트 디렉토리에서 실행합니다.

```bash
cd /home/mind4u/dev/minimind/minimind-j2
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

또는 기존 `../minimind/.venv/bin/python`을 사용할 수 있습니다. 백본이 사용하는 Transformers 4 API에 맞춰 `<5`로 제한했습니다. 모델·데이터를 자동 다운로드하지 않습니다. `prepare` 명령만 명시적으로 외부 데이터셋을 내려받습니다.

## 입력 형식

```json
{"question":"What do people use to write?","A":"Spoon","B":"Pen","C":"Chair","D":"Shoe","E":"Plate","answer":"B"}
```

JSON 배열 또는 JSONL을 지원합니다. `question`, A–E는 비어 있지 않은 문자열이어야 합니다. 학습·평가·보정에는 `answer`가 필수이며, 추론에서는 생략합니다. 정답 필드는 프롬프트에 넣지 않습니다. 선택지 라벨이 고유한 단일 토큰인지 검사합니다.

MiniMind chat template에서 thinking을 끄고 assistant 응답 시작 위치를 사용합니다. 오른쪽 패딩 후 각 샘플의 마지막 유효 토큰을 선택하므로 RoPE 위치가 배치 길이에 따라 바뀌지 않습니다. `--max-length`를 초과하면 오류를 내며 선택지를 조용히 잘라내지 않습니다.

## 데이터 준비

```bash
.venv/bin/python jev.py prepare --output dataset --seed 0
```

`tau/commonsense_qa`의 train을 `train.jsonl`로 변환합니다. 공식 validation은 섞은 후 절반씩 `calibration.jsonl`과 `eval.jsonl`로 나눕니다. 정답이 공개되지 않은 공식 test는 사용하지 않습니다. 직접 준비한 데이터도 같은 형식으로 사용할 수 있으며 학습·보정·평가 예제를 서로 분리해야 합니다. 출력 폴더는 비어 있어야 합니다.

## 기본 모델 평가와 head 학습

먼저 실제 native MiniMind state dict와 정확히 일치하는 설정을 준비합니다. 다음 가중치 경로는 예시이며 해당 파일의 존재나 사전 학습 품질을 보장하지 않습니다.

```bash
.venv/bin/python jev.py evaluate \
  --weights ../minimind/out/full_sft_768.pth \
  --config configs/minimind_768.json --data dataset/eval.jsonl

.venv/bin/python jev.py train \
  --weights ../minimind/out/full_sft_768.pth \
  --config configs/minimind_768.json \
  --data dataset/train.jsonl --output checkpoint \
  --epochs 3 --batch-size 16 --learning-rate 1e-4 --seed 0
```

가중치는 원본 MiniMind trainer가 저장하는 native state dict 형식입니다. optimizer 등을 포함한 resume 컨테이너나 Qwen 형식 HF export를 직접 읽는 경로는 제공하지 않습니다. 다른 hidden size, 층 수, MoE, 어휘 크기를 사용했다면 해당 설정 JSON과 `--tokenizer PATH`를 전달하세요. 토크나이저 어휘 수 검사만으로 토큰 ID 의미의 일치를 증명할 수 없으므로 **가중치를 학습할 때 사용한 토크나이저**가 필요합니다. 상태 사전은 strict 모드로 읽습니다.

CUDA가 있으면 사용하고 그렇지 않으면 CPU에서 실행합니다. `--device cpu`, `--max-length 2048`, `--limit 100`으로 실행 범위를 조절할 수 있습니다. 모델은 float32로 로드하며 head만 역전파합니다. 큰 배치에서 메모리가 부족하면 배치 크기와 입력 길이를 줄이세요.

결과 폴더에는 `model.pt`(본체와 분리된 head 전체), `config.json`, 토크나이저 파일, `jev.json`(선택지 ID, 길이, 학습 설정·손실·데이터 해시)을 저장합니다. 기존 결과를 덮어쓰지 않습니다. `train --checkpoint checkpoint --output checkpoint-next ...`는 저장된 가중치에서 새 optimizer로 추가 학습하며 optimizer 재개 기능은 아닙니다.

## 확률 보정·평가·추론

```bash
.venv/bin/python jev.py calibrate \
  --checkpoint checkpoint --data dataset/calibration.jsonl

.venv/bin/python jev.py evaluate \
  --checkpoint checkpoint --data dataset/eval.jsonl

.venv/bin/python jev.py predict \
  --checkpoint checkpoint --data data/input.json
```

보정은 고정된 로짓의 NLL을 최소화하는 단일 양수 temperature를 찾습니다. 역온도에 대한 볼록 목적함수를 구간 탐색하며 T 범위는 `[0.01, 1000]`입니다. T=1도 후보로 포함합니다. `temperature.json`에 보정 전후 지표와 모델·데이터 해시를 저장하며, 같은 체크포인트에 다시 실행하면 보정 파일을 교체합니다. 모델 해시가 바뀌면 기존 온도 사용을 거부합니다. 양수 temperature는 argmax를 변경하지 않습니다.

평가·추론은 저장된 온도를 자동 적용합니다. `--temperature 1`로 미보정 결과를 비교할 수 있습니다. 평가는 accuracy, NLL, 15-bin ECE, multiclass Brier score(클래스별 제곱 오차의 합)를 JSON으로 출력합니다. 추론은 질문별 JSONL로 선택된 라벨·선택지 텍스트·A–E 확률을 출력합니다.

확률은 **주어진 다섯 후보 사이의 조건부 확률**이며, 정답이라는 보장이나 도메인 전반에 통용되는 신뢰도가 아닙니다. 보정 데이터의 NLL 개선은 독립 평가 성능 개선을 보장하지 않습니다. MiniMind의 영어 상식 질문 능력은 실제 학습 가중치와 평가 결과로 확인해야 합니다.

## 테스트

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

다운로드 없이 로컬 토크나이저와 무작위 초기화한 32차원/1층 모델을 사용합니다. 테스트는 본체 고정·손실 감소·배치 일치·정답 누출 방지·저장 왕복·temperature 성질·CLI 흐름을 검사합니다. 학습된 Jev의 상식 판단 성능 검증은 별도입니다.

## 출처

구조 참고: `/home/mind4u/dev/build-your-own-jev/scripts/{finetune-lm-head,predict-lm-head,eval-lm-head,calibrate-lm-head,prepare_dataset}.py`. 원본 Qwen 다운로드와 HF 저장 흐름을 MiniMind native 모델로 다시 구현했습니다. 백본·토크나이저 출처는 `../minimind/model/`이며 MiniMind MIT 라이선스를 `LICENSE`에 포함했습니다. 원본 참고 프로젝트의 스크립트는 복사하지 않았습니다.
