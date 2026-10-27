# 에이전트용 저장소 안내

이 파일은 저장소 전체의 탐색 안내입니다. 트리는 주요 경로만 표시합니다.
작업 대상 프로젝트의 README와 실제 코드를 함께 확인하고, 하위 `AGENTS.md`가 생기면
해당 경로에 적용되는 추가 지침도 읽으세요.

## 전체 구조

```text
./
├── AGENTS.md                 # 전체 프로젝트 구조와 주요 파일 안내
├── outputs/                   # 테스트 결과, 실행 로그와 생성 산출물
├── reports/                   # 테스트·검증·분석 보고서
├── test_scenarios/            # 모델별 테스트 입력·절차·판정 기준 (필요 시 생성)
├── gpt/                      # GPT 단계별 노트북 실습과 모델 구조 개선 실험
├── unsloth/                  # Unsloth 기반 Qwen 학습·추론·배포 노트북 실습
├── minimind/                 # 기본 자기회귀 언어 모델: Dense/MoE, SFT와 RL
├── minimind-v/               # 이미지 이해를 추가한 시각·언어 모델(VLM)
├── minimind-o/               # 텍스트·이미지·음성을 처리하는 Omni 모델
├── minimind-a/               # 이미지·지시문·상태에서 행동 청크를 생성하는 VLA
├── minimind-j/               # 한 입력에 대한 여러 타입별 질문을 병렬 처리하는 판단 모델
├── minimind-d/               # 마스킹과 반복 복원 기반 확산 언어 모델
└── minimind-la/              # 선형 어텐션과 전체 어텐션을 결합하는 모델 실험
```

- 각 프로젝트는 별도의 실행 디렉토리입니다. `model`, `dataset`, `trainer` 같은 이름이
  겹치므로 프로젝트 경계를 구분하고, 해당 프로젝트 디렉토리를 기준으로 실행하세요.
- `model_minimind.py`는 여러 프로젝트에 별도 사본으로 존재합니다. 하나를 수정한다고
  다른 프로젝트에 반영되지 않습니다. 공통 변경이 필요한지 검토하되 프로젝트별 차이를
  확인하고, 작업 범위에 해당하는 파일만 수정하세요.
- 모델 가중치, 데이터, `out/`, `checkpoints/` 등 실행 산출물은 내려받거나 실행해야
  생길 수 있습니다. README에 경로가 등장해도 로컬에 존재한다고 가정하지 마세요.
- `.git/`은 버전 관리, `.agents/`, `.codex/`, `.aws/`는 환경 설정 경로입니다.
  일반 소스 수정 대상으로 취급하지 마세요. 캐시 디렉토리도 구조 설명에서 생략합니다.

## gpt — GPT 실습과 모델 구조 실험

```text
gpt/
├── README.md                 # uv 환경 설치와 노트북 실행 안내
├── pyproject.toml / uv.lock  # Python 3.12 프로젝트 의존성과 잠금 파일
├── Chapter_*.ipynb           # 벡터·데이터·Attention·GPT·사전 학습·SFT·LoRA·DPO 실습
├── previous_chapters.py      # 노트북에서 재사용하는 GPT 모델·학습·생성·다운로드 함수
├── model_improvements/       # KV cache, GQA, MLA, MoE, DeltaNet 등 구조 실험
└── .tours/                   # 모델 개선 코드 탐색용 CodeTour
```

`gpt/README.md`와 대상 노트북, 가져오는 `previous_chapters.py` 또는
`model_improvements/` 모듈을 함께 확인하세요. `previous_chapters.py`를 수정하면
이를 공유하는 여러 노트북에 영향을 줄 수 있습니다. 원본의 저작권·라이선스 안내를 유지합니다.
설치는 `gpt/`에서 `uv sync --locked`, 실행은 `uv run jupyter lab`을 사용하고
노트북 커널은 `gpt/.venv/bin/python`을 선택합니다. 데이터·모델 가중치의 존재와
다운로드 경로를 확인하고, 학습 셀의 실제 실행과 구문 검사만 수행한 경우를 구분하세요.
새 실행 산출물은 `outputs/gpt/<실행 ID>/`, 보고서는 `reports/gpt/<실행 ID>.md`에 저장합니다.

## unsloth — Unsloth 노트북 실습

```text
unsloth/
├── README.md                 # 환경별 설치·실행과 검증 범위 안내
├── pyproject.toml / uv.lock  # 기본 의존성과 grpo·qat·phone·tools 추가 옵션
├── .python-version          # Python 버전 지정
├── Qwen*.ipynb               # Decision, Vision, GRPO, 증류, 도구 호출, QAT, Phone 실습
└── scripts/
    ├── execute_notebook.py   # 셀별 실행 기록, executed.ipynb와 status.json 저장
    └── run_notebook_on_pod.sh # 기존 Pod 내부에서 uv 환경 준비와 노트북 실행
```

`unsloth/README.md` → `pyproject.toml` → 대상 노트북 순서로 확인하세요.
Python 3.12 / Linux x86_64 환경을 사용하며, 학습에는 호환되는 NVIDIA GPU와
CUDA 드라이버가 필요합니다. `unsloth/`에서 `uv sync --locked`로 설치하고
실습에 따라 `--extra grpo`, `--extra qat`, `--extra phone`, `--extra tools`를 선택합니다.
`grpo`, `qat`, `phone`은 하나씩 선택하고 `--all-extras`는 사용하지 않습니다.
환경 변경 전 커널을 종료하고 설치 후 다시 시작하세요.

패키지 설치는 uv로 관리하고, 노트북에 설치 셀을 추가하거나 `--no-deps`로
의존성을 우회하지 마세요. 배포용 원본 노트북에는 실행 출력과 실행 번호를 남기지 않고,
실행 사본·로그·지표는 `outputs/unsloth/<실행 ID>/`, 보고서는
`reports/unsloth/<실행 ID>.md`에 저장합니다. `execute_notebook.py`의
`--output-dir`도 이 규칙에 맞춥니다. Pod 실행 스크립트는 원격 작업 경로를 사용하므로
결과를 내려받아 같은 규칙으로 정리하세요. 문서 확인만으로 Pod 실행이나 학습을 시작하지 마세요.

## minimind — 기본 언어 모델

```text
minimind/
├── README.md / README_en.md / README_ko.md
├── requirements.txt             # 프로젝트 의존성
├── eval_llm.py                  # CLI 추론: native PyTorch 및 Transformers 모델
├── model/
│   ├── model_minimind.py         # 설정, RMSNorm, RoPE, Attention, MLP/MoE, LM 및 생성
│   └── model_lora.py             # LoRA 적용·저장·로드·병합
├── dataset/
│   └── lm_dataset.py            # 사전 학습·SFT·선호 학습 등의 데이터 처리
├── trainer/
│   ├── train_tokenizer.py       # 토크나이저 학습 및 입력 텍스트 읽기
│   ├── test_train_tokenizer.py  # 토크나이저 데이터 로딩·학습 관련 테스트
│   ├── train_pretrain.py        # 다음 토큰 예측 사전 학습
│   ├── train_full_sft.py        # 전체 매개변수 지도 미세 조정
│   ├── train_lora.py            # LoRA 미세 조정
│   ├── train_distillation.py    # 교사·학생 모델을 사용하는 지식 증류
│   ├── train_dpo.py             # 선호 응답 쌍을 사용하는 DPO
│   ├── train_ppo.py             # Actor/Critic 기반 PPO
│   ├── train_grpo.py            # 그룹 상대 보상 기반 GRPO 및 CISPO 손실
│   ├── train_agent.py           # 여러 턴 도구 실행과 보상을 사용하는 Agentic RL
│   ├── rollout_engine.py        # PyTorch/SGLang rollout 엔진
│   └── trainer_utils.py         # 학습 공통 기능, 체크포인트 등
├── scripts/
│   ├── convert_model.py         # 가중치 형식 변환 및 LoRA 병합 관련 기능
│   ├── serve_openai_api.py      # OpenAI API 형식 호환 서버
│   ├── chat_api.py              # API 호출 예제
│   ├── web_demo.py              # Streamlit 채팅 화면
│   ├── eval_toolcall.py         # 도구 호출 평가
│   └── runpod_*/                # 작업별 클라우드 학습·준비·평가 스크립트
└── images/                     # README 그림과 학습 곡선
```

모델 연산을 변경하면 `model/model_minimind.py`와 관련 학습·평가 경로를 확인합니다.
입력 형식이나 마스킹 변경은 `dataset/lm_dataset.py`도 함께 확인합니다.
RL rollout 변경은 `trainer/rollout_engine.py`와 사용하는 학습 스크립트를 같이 봅니다.
`scripts/runpod_grpo/` 등에는 별도 구현이 있으므로 기본 trainer와 동일하다고 가정하지 마세요.

## minimind-v — 시각·언어 모델

```text
minimind-v/
├── README.md / README_en.md / README_ko.md
├── requirements.txt
├── eval_vlm.py                  # 이미지 질의·응답 추론
├── model/
│   ├── model_minimind.py        # 언어 백본
│   └── model_vlm.py             # 비전 인코더, Projector, 시각 임베딩 결합
├── dataset/
│   ├── lm_dataset.py           # 이미지·대화 데이터 로딩과 전처리
│   └── eval_images/             # 이미지 평가 예제
├── trainer/
│   ├── train_pretrain_vlm.py    # Projector 사전 정렬
│   ├── train_sft_vlm.py         # 시각 지시 미세 조정
│   └── trainer_utils.py         # 모델 초기화, 고정 전략, 저장·재개 등
├── scripts/
│   ├── convert_vlm.py          # VLM 가중치 변환
│   ├── web_demo_vlm.py         # Gradio 웹 데모
│   └── runpod_sharegpt4v/      # ShareGPT4V 클라우드 학습·검증 확장
└── images/                    # 구조도와 실험 자료
```

이미지 특징의 shape, 자리표시자 수, Projector를 바꾸면 모델과 데이터 처리를 함께
확인합니다. 비전 인코더 설정은 사용하는 가중치·processor와 일치해야 합니다.
학습 가능한 층의 범위는 `freeze_llm` 및 trainer의 고정 전략을 확인하세요.

## minimind-o — 음성·시각·언어 통합 모델

```text
minimind-o/
├── README.md / README_en.md / README_ko.md
├── requirements.txt
├── eval_omni.py                 # Omni 명령행 추론
├── model/
│   ├── model_minimind.py        # 기본 언어 Transformer
│   ├── model_omni.py            # Thinker/Talker, 모달 결합, 오디오 생성 및 VAD
│   ├── speaker/                # 화자 조건 관련 리소스
│   └── vad/                    # 발화 감지 관련 리소스
├── dataset/
│   ├── omni_dataset.py         # 음성·이미지·텍스트 샘플과 시퀀스 처리
│   └── eval_omni/              # 추론·평가 입력 예제
├── trainer/
│   ├── train_sft_omni.py       # T2A/A2A/I2T 학습 진입점
│   ├── train.sh               # mini 실행 명령 및 주석 처리된 full 학습 예제
│   └── trainer_utils.py        # 초기화, 학습 모드, 저장·복원 등
├── scripts/
│   ├── convert_omni.py         # Omni 가중치 변환
│   ├── web_demo_omni.py        # 비실시간 Gradio 데모
│   └── runpod_voiceassistant/  # 음성 데이터·코덱·클라우드 학습·추론 검증
├── webui/
│   ├── web_demo.py            # 실시간 음성 상호작용 서버
│   └── web_demo.html          # 실시간 화면
└── images/                   # 구조도, 시퀀스 설명과 실험 자료
```

오디오 코드북, 지연 스케줄, 조건 토큰 또는 샘플링 주파수를 바꾸면
`model_omni.py`, `omni_dataset.py`, 추론 경로를 함께 확인합니다.
`all`, `audio_proj`, `vision_proj` 학습 모드는 업데이트 대상이 다릅니다.
Gradio 데모와 실시간 `webui/` 경로를 구분하세요.

## minimind-a — 행동 생성 모델

```text
minimind-a/
├── README.md                   # VLA 설명, 입력·출력 의미와 실행 방법
├── requirements.txt / requirements-dev.txt
├── pytest.ini                  # 프로젝트 기준 테스트 경로
├── train_vla.py                # VLA 학습·검증·체크포인트
├── eval_vla.py                 # 저장 모델의 행동 청크 추론
├── model/
│   ├── model_vla.py            # 문맥 인코딩, 행동 Transformer, flow matching, Euler 적분
│   ├── model_vlm.py            # MiniMind-V 기반 시각·언어 구성
│   └── model_minimind.py       # 언어 백본
├── dataset/vla_dataset.py      # 이미지·지시문·상태·행동 로딩 및 배치 구성
├── configs/minimind_v_768.json # 기존 MiniMind-V 가중치 재사용 설정
├── tests/                     # 모델·데이터·학습·SO101 데이터 테스트
├── scripts/
│   ├── make_toy_data.py        # 합성 행동 데이터 생성
│   └── runpod_so101/           # SO101 데이터·학습·시뮬레이터 rollout 확장
└── docs/                      # 구조도, 출처 기록, 검증 결과와 설계 문서
```

읽는 순서는 `model/model_vla.py` → `dataset/vla_dataset.py` → `train_vla.py`
→ `eval_vla.py`를 권장합니다. 핵심은 `[B,H,A]` 행동 청크와 flow matching입니다.
비전 경로는 MiniMind-V의 SigLIP을 사용하며 외부 인코더 파일이 필요합니다.
기존 VLM을 재사용할 때 설정의 레이어 수는 원본 모델 기준이고, 실제 VLA는 앞쪽
절반 레이어를 사용하므로 README와 로딩 코드를 확인하세요.

## minimind-j — 병렬 타입별 판단 모델

```text
minimind-j/
├── README.md                   # 타입별 판단 형식과 병렬 처리 원리
├── requirements.txt / pytest.ini
├── dataset.py                 # 데이터 검증, 토크나이저, 질문 packing과 배치 처리
├── train.py                   # 판단 모델 학습·검증
├── decide.py                  # 요청 JSON 또는 CLI 입력을 받아 판단 결과 출력
├── metrics.py                 # 정확도, KL, Brier, ECE 및 그룹별 지표
├── model/
│   ├── decision.py            # 기존 LM head를 이용한 질문별 후보 점수 계산
│   └── model_minimind.py       # block attention 및 위치 입력을 지원하는 백본
├── data/                      # 학습·검증 JSONL, 추론 요청, 구 단일 선택 예제
├── tests/                     # 판단, 타입별 학습, 강건성 테스트
├── scripts/
│   ├── evaluate_robustness.py  # 순서·표현 변화 등에 대한 강건성 평가
│   └── runpod_typed/           # 외부 데이터 클라우드 학습·보고
└── docs/                      # 설계, 검증 기록과 smoke 결과
```

읽는 순서는 `dataset.py` → `model/decision.py` → `train.py` → `decide.py`입니다.
`choice`는 문자열 선택, `score`는 유한한 정수 등급 분류, `noul`은 boolean이며
JSON null과 다릅니다. 별도 decision MLP 대신 기존 LM head를 사용합니다.
질문 packing을 바꾸면 block causal mask, 질문별 RoPE 위치, 정답의 입력 누출 여부와
타입 검증을 함께 확인하세요. 후보 내 확률을 보정된 신뢰도로 해석하지 마세요.

## minimind-d / minimind-la — 모델 구조 실험

```text
minimind-d/
├── model/model_minimind_dllm.py # 비인과 Attention, 마스킹 학습, 반복 복원 생성
├── trainer/train_dllm.py       # 확산 언어 모델 학습
├── eval_dllm.py                # 확산 모델 평가·대화
└── scripts/
    ├── chat_dllm.py            # 대화 스크립트
    └── web_dllm.py             # 웹 데모

minimind-la/
├── model/model_minimind_linear.py # GatedDeltaNet 기반 선형/전체 Attention 혼합
└── run_linear.py                  # 기본 모델 모듈을 교체해 대상 스크립트를 실행
```

DLLM은 `model.model_minimind`의 기본 구현을 가져오지만 해당 디렉토리에는 기본
모델 파일이 없습니다. 실행 전 import 경로와 기본 모델 의존성을 확인하세요. 두 실험
디렉토리에는 현재 별도 README가 없습니다. 선형 어텐션의 실행기는 `sys.modules`를
교체하고 대상 스크립트 디렉토리로 이동하므로 일반 학습 진입점과 실행 방식이 다릅니다.
선형 모델은 선택적 `causal_conv1d`, `flash-linear-attention` 가속이 없으면 PyTorch
경로로 폴백합니다.

## 수정 후 확인할 항목

- 테스트는 프로젝트 디렉토리에서 실행합니다. 전체 저장소 테스트를 한 번에 모으면
  동명 모듈 때문에 import 충돌이 발생할 수 있습니다.
- `minimind-a/`, `minimind-j/`: 각 디렉토리에서 `python -m pytest -q`.
- `minimind/`: 프로젝트 디렉토리에서
  `python -m unittest discover -s trainer -p 'test_train_tokenizer.py'`.
- 클라우드 확장은 `scripts/runpod_*/README.md`와 해당 테스트 안내를 먼저 확인합니다.
  경로 탐색이나 문서 확인만으로 학습·리소스 생성 스크립트를 실행하지 마세요.
- 문서 변경은 상대 링크와 명령의 실행 위치를 확인합니다. 모델 변경은 관련 테스트와
  가능한 범위의 작은 순전파·역전파 검증을 사용하고, 실제 학습이 필요한 검증과 구분합니다.
- 체크포인트·모델 성공 여부는 소스 파일이나 과거 검증 문서의 존재만으로 판단하지
  마세요. 현재 파일, 실행 결과와 필요한 외부 리소스를 확인하고 검증 범위를 보고합니다.

## 테스트 결과와 보고서 저장 규칙

- 각종 테스트 결과와 실행 산출물은 저장소 루트의
  `outputs/<프로젝트>/<실행 ID>/` 아래에 정리합니다. 단위·통합·수동 테스트,
  노트북 실행과 클라우드 검증 모두 같은 규칙을 적용합니다.
- 실행 ID는 `20261027-153000-smoke`처럼 날짜·시간과 작업명을 조합해
  실행별로 구분합니다. 로그, 지표 JSON/CSV, 생성 샘플, 실행된 노트북,
  이미지 및 리소스 사용 기록은 해당 실행 폴더에 함께 보관합니다.
- 결과를 설명하는 테스트·검증·분석 보고서는 저장소 루트의
  `reports/<프로젝트>/<실행 ID>.md`에 저장합니다. 보고서에는 실행 명령과
  작업 디렉토리, 환경·패키지 버전, 모델·데이터 리비전, 판정 근거,
  실패·미실행 항목과 검증 한계를 기록하고 대응하는 `outputs/` 경로를 명시합니다.
- 프로젝트 디렉토리에서 실행하더라도 저장 경로는 저장소 루트를 기준으로
  지정합니다. 필요한 폴더는 생성하고, 스크립트의 출력 옵션으로 경로를 설정하세요.
- `test_scenarios/`에는 재사용할 입력·절차·판정 기준을 두고,
  실제 실행 결과는 `outputs/`, 결과 보고서는 `reports/`에 저장합니다.
  기존 기록을 옮길 때는 참조 링크도 함께 갱신합니다.
- `outputs/`는 `.gitignore` 대상입니다. 보고서에는 핵심 결과를 직접 요약하고,
  산출물을 별도로 보관했다면 재확인할 수 있는 위치도 기록합니다.
  인증 정보나 비밀 값은 산출물과 보고서에 포함하지 마세요.

## 모델 테스트 시나리오

`test_scenarios/<모델>/`에 모델별 수동·통합 테스트 입력·절차·판정 기준을 작성합니다.
해당 경로에 README가 있으면 먼저 확인하고, 폴더가 없으면 필요할 때 생성하세요.
가중치·입력·생성 설정을 명시하고, 실행 결과와 판정 근거는 위 저장 규칙에 따라
`outputs/`와 `reports/`에 기록합니다. 미실행 시나리오를 통과 결과로 보고하지 마세요.
자동화된 단위 테스트는 각 프로젝트의 기존 테스트 위치에 유지합니다.
