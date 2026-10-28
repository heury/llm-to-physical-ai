# MiniMind SFT 데이터 확인 프롬프트

## 요청

`caspar/MiniMind-Mixed-SFT-Training-v1.0`의 실제 샘플을 보여주고, 현재 저장소의 `SFTDataset`으로 변환하면 어떤 데이터가 되는지 설명해 주세요. 원본 대화 → JSONL 입력 → 채팅 템플릿 → 토큰 → 학습 라벨 → 모델 손실 계산의 연결을 실제 실행 결과로 보여 주세요.

## 확인할 코드와 데이터

- `minimind/README.md` 또는 `minimind/README_ko.md`
- `minimind/dataset/lm_dataset.py`: `pre_processing_chat`, `post_processing_chat`, `SFTDataset`
- `minimind/trainer/train_full_sft.py`: 데이터셋 생성, 배치 로딩, 모델 호출
- `minimind/model/model_minimind.py`: causal LM의 labels 이동과 손실 계산
- `minimind/model/`의 실제 로컬 토크나이저
- 데이터셋: https://huggingface.co/datasets/caspar/MiniMind-Mixed-SFT-Training-v1.0

## 대화에서 확인한 내용

이 데이터셋은 영어 대화 SFT 자료로, 공개 설명 기준 학습 대화는 다음과 같습니다.

| 출처 | 대화 수 |
| --- | ---: |
| GSM8K | 6,851 |
| UltraChat | 207,865 |
| SmolTalk | 1,020,912 |
| 합계 | 1,235,628 |

설정 이름은 `conversations`입니다. 행에는 `dataset`, `split`, `index`, `first_window`, `windows`, `assistant_targets`, `eos_targets`, `source_component`, `messages_json` 등이 있습니다. `messages_json`은 대화 배열을 직렬화한 문자열이며 `json.loads()`로 풀어야 합니다.

기본 `SFTDataset`은 이 행을 그대로 받는 대신 다음 JSONL 형식을 읽습니다.

```json
{"conversations": [{"role": "user", "content": "질문"}, {"role": "assistant", "content": "답변"}]}
```

### 실제 첫 샘플

출처는 train의 첫 행이며, 대화 중 API로 확인한 저장소 리비전은 `542384d27d374d668413d8ade5b9973b7b063bee`입니다.

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nSolve step by step. End your answer with #### followed by the final number."
    },
    {
      "role": "assistant",
      "content": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
    }
  ]
}
```

### 이전 로컬 변환 결과

기본 `SFTDataset`과 로컬 `minimind/model/` 토크나이저로 `max_length=768`, Python `random.seed(0)`을 사용해 실행했습니다. 이 실행에서는 system 메시지가 추가되지 않았고 빈 think 블록은 제거되었습니다.

```text
<|im_start|>user
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Solve step by step. End your answer with #### followed by the final number.<|im_end|>
<|im_start|>assistant
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72<|im_end|>
```

- 반환값은 `input_ids`, `labels` 두 개의 `torch.int64` 텐서이며 각각 shape `[768]`입니다.
- 유효 입력 149토큰, 패딩 619토큰입니다.
- 학습 라벨은 0부터 세는 위치 81~148의 68토큰입니다. assistant 답변과 `<|im_end|>`, 뒤의 개행을 포함합니다.
- 사용자 질문, assistant 헤더, 패딩의 라벨은 `-100`입니다.
- 확인된 특수 토큰 ID: `<|im_start|>`는 1, `<|im_end|>`는 2, 패딩 `<|endoftext|>`는 0입니다.
- 모델 내부에서 `logits[..., :-1, :]`와 `labels[..., 1:]`을 대응시킵니다. 데이터셋에서 라벨을 미리 한 칸 이동하지 않습니다.

기존 산출물과 보고서는 저장소 루트 기준 다음 경로에 있습니다.

- `outputs/minimind/20261029-sftdataset-preview/preview.py`
- `outputs/minimind/20261029-sftdataset-preview/sample.jsonl`
- `outputs/minimind/20261029-sftdataset-preview/preview.json`: 전체 input_ids와 labels 포함
- `outputs/minimind/20261029-sftdataset-preview/preview.log`
- `reports/minimind/20261029-sftdataset-preview.md`

`outputs/`는 gitignore 대상이므로 다른 환경에서 이 파일들의 존재를 가정하지 마세요. 위 수치는 과거 실행 결과이며 현재 코드와 토크나이저로 다시 확인하기 전에는 새 검증 결과로 제시하지 마세요.

## 수행할 작업

1. 현재 코드와 로컬 파일을 확인하고, 기존 산출물이 있다면 먼저 활용하세요.
2. 원본 데이터가 필요하면 전체 다운로드 대신 공개 API 등으로 필요한 소수 행만 확인하세요. 실제 샘플과 설명용 가상 예시, 생략한 부분을 구분하세요.
3. 원본 `messages_json`을 `conversations` JSONL로 변환하는 과정을 보여 주세요.
4. 실제 `SFTDataset`으로 샘플을 처리하고 seed와 max_length를 기록하세요. 무작위 system 추가와 빈 think 처리로 결과가 달라질 수 있음을 설명하세요.
5. 디코딩한 입력, 전체 input_ids와 labels, 유효 길이, 패딩 수, 학습 대상 수를 저장하세요. 채팅 답변에서는 핵심 구간만 보여 주세요.
6. 사용자/assistant 경계와 EOS/패딩 경계의 토큰을 `위치 / 디코딩한 토큰 / input_id / label` 표로 보여 주세요.
7. `-100`은 손실 계산 제외를 뜻하며, 입력에서 해당 토큰을 제거하거나 attention을 가리는 값은 아니라는 점을 설명하세요.
8. `train_full_sft.py`가 이 텐서 쌍을 어떻게 배치로 받아 모델에 전달하는지 설명하세요.
9. 기본 SFTDataset의 절단·패딩과 공개 Mixed SFT 실험의 길이 340, stride 256 윈도 처리를 구분하세요. 기본 변환 예시를 배포된 바이너리 윈도의 완전 재현이라고 주장하지 마세요.

## 실행 범위와 기록

- 데이터 확인과 전처리만 수행하며 학습이나 클라우드 리소스 생성을 시작하지 마세요.
- 프로젝트 코드를 수정할 필요가 없다면 수정하지 마세요.
- 실행 산출물은 `outputs/minimind/<실행 ID>/`, 보고서는 `reports/minimind/<실행 ID>.md`에 저장하세요. 경로는 저장소 루트를 기준으로 합니다.
- 보고서에 작업 디렉토리, 실행 명령, 환경·패키지 버전, 데이터 리비전, 토크나이저 출처, seed, 실행 여부와 한계를 기록하세요.
- 문서·코드 확인, 전처리 실행, 모델 forward, 실제 학습을 구분해 보고하세요.
