# minimind-la


## uv 환경

각 모델 디렉토리는 독립 uv 프로젝트입니다. Python은 `.python-version`에
3.12.15로 고정하고, 의존성은 `pyproject.toml`과 `uv.lock`으로 관리합니다.

```bash
cd minimind-la
uv sync --locked
source .venv/bin/activate
# 또는 uv run python <스크립트>로 실행
```

`uv sync`는 Python 3.12.15가 없으면 다운로드를 시도합니다. 사용 중인 uv가
해당 버전 다운로드를 지원하지 않으면 uv를 업데이트하거나 Python 3.12.15를
설치한 뒤 `uv sync --python /path/to/python3.12 --locked`를 실행하세요.
기존 `requirements*.txt`는 이전 pip 실행 환경을 위한 참고 자료이며,
uv 환경의 의존성을 변경할 때는 `uv add` / `uv remove`를 사용하세요.

기본 MiniMind 스크립트와 함께 사용하는 실험이므로 `minimind`의 의존성을
기준으로 구성했습니다. 선택적 CUDA 가속 패키지는 포함하지 않습니다.

이 설정은 외부 패키지 환경을 구성합니다. 기존 소스의 로컬 import 경로는
별도 확인이 필요합니다. `run_linear.py`는 모델 모듈을 교체하고 대상 스크립트
디렉토리로 이동하므로, 대상의 로컬 모듈 경로도 필요합니다.
형제 프로젝트의 `model/__init__.py` 때문에 단순히 `PYTHONPATH`를 추가하는
것만으로는 모델 네임스페이스가 합쳐지지 않습니다. 실행 경로 수정은 이 환경
설정에 포함하지 않았습니다. 작은 CPU 모델의 순전파·역전파만 검증했으며,
CLI 전체 실행과 실제 학습은 검증하지 않았습니다.


### 의존성 업데이트

일반 패키지는 버전 고정을 제거했습니다. 모델 API 호환성을 위해
`transformers>=4.57.6,<5`, `torch>=2.6,<3` 범위는 유지합니다.
TRL은 기존 API 계열인 `>=0.13,<0.14`를 유지합니다.

`uv.lock`은 설치 버전을 고정합니다. `uv sync --locked`는 최신 버전을
조회하는 명령이 아닙니다. 허용 범위 내 최신 버전으로 갱신하려면 이
프로젝트 디렉토리에서 다음을 실행하고 테스트를 다시 확인하세요.

```bash
uv lock --upgrade
uv sync --locked
```

Python 3.12.15 고정은 유지합니다. torch와 관련 vision/audio 패키지는
의존성 해결기가 선택한 조합을 잠금 파일에 함께 기록합니다.
