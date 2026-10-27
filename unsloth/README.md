# Unsloth 실습

Python 3.12 / Linux x86_64 환경에서 Unsloth 노트북을 실행하는 uv 프로젝트입니다.
학습 의존성을 `pyproject.toml`에서 관리하며, 노트북 원본의 설치 버전 고정은 제거했습니다.
각 노트북은 Installation 셀과 그 위의 소개·뉴스를 제거하고 실습 본문부터 시작합니다.
패키지 설치는 아래 uv 명령으로 터미널에서 진행합니다.

## 설치 및 실행

아래 명령은 이 `unsloth/` 디렉토리에서 실행합니다.

```bash
# A40에서 검증한 잠금 파일의 패키지 조합으로 설치
uv sync --locked
uv run jupyter lab
```

VS Code에서는 `.venv/bin/python`을 노트북 커널로 선택합니다.
잠금 파일은 CUDA 13.0용 패키지를 사용하므로, GPU 실행에는 이를 지원하는 NVIDIA
드라이버가 필요합니다. CPU만 있는 환경에서 학습 노트북을 실행할 수는 없습니다.
일반 Decision, Vision, Vision GRPO, Knowledge Distillation 실습은 기본 환경을 사용합니다.
다음 실습은 추가 패키지를 선택하세요.

| 실습 | 검증한 환경 설치 | 실행 |
| --- | --- | --- |
| Qwen3 GRPO (vLLM) | `uv sync --locked --extra grpo` | `uv run --locked --extra grpo jupyter lab` |
| QAT | `uv sync --locked --extra qat` | `uv run --locked --extra qat jupyter lab` |
| Phone deployment | `uv sync --locked --extra phone` | `uv run --locked --extra phone jupyter lab` |
| Tool calling | `uv sync --locked --extra tools` | `uv run --locked --extra tools jupyter lab` |

`grpo`, `qat`, `phone`은 환경을 전환하며 하나씩 선택합니다. `--all-extras`는 사용하지 않습니다.
`tools`는 다른 옵션과 함께 선택할 수 있습니다.
환경을 바꾸기 전에는 실행 중인 커널을 종료하고, 설치 후 커널을 다시 시작하세요.
동시에 여러 환경이 필요하면 별도의 가상환경을 사용합니다.
배포용 노트북의 저장된 출력과 실행 번호는 제거했습니다. 테스트 실행 결과는 원본
노트북에 덮어쓰지 않고 별도 검증 기록에 보관합니다.

## 최신 버전과 재현성

`uv sync --upgrade`는 새로운 버전을 다시 조회합니다.
`uv sync --locked`는 저장된 `uv.lock` 조합을 재현하며, 추가 옵션도 동일하게 지정합니다.
패키지 상호 의존성은 유지합니다. `--no-deps`나 의존성 강제 덮어쓰기는 사용하지 않습니다.

2026-10-27 A40 검증 환경:

- Unsloth / Unsloth Zoo: 2026.10.9
- 기본 환경: Transformers 5.17.0, TRL 1.15.0, PyTorch 2.14.1
- GRPO: vLLM 0.32.0이 요구하는 PyTorch 2.13.0 사용
- QAT: MSLK 1.3.0의 바이너리 호환성에 맞춰 PyTorch 2.13.0 사용
- Phone: 노트북에서 사용하는 ExecuTorch를 직접 설치하며 Transformers 5.17.0 사용

PyPI에는 더 최신 Transformers / TRL도 있지만 Unsloth의 상한 제약이 있어
동시에 선택할 수 없습니다. Phone에서 사용하지 않는 `optimum-executorch`는 제거했습니다.
이 패키지가 요구하는 Transformers 5.0.0rc1은 최신 TorchAO의 저장 API와 충돌했습니다.
선택된 버전은 `uv.lock`에서 확인할 수 있습니다.
`xformers`와 QAT의 `mslk`는 PyTorch 공식 CUDA 13 인덱스에서 설치합니다.
PyPI의 CUDA 12.8용 `xformers`를 설치한 기존 환경은 같은 버전 번호 때문에
일반 동기화만으로 교체되지 않을 수 있으므로, 한 번 다음과 같이 재설치하세요.

```bash
uv sync --locked --reinstall-package xformers
# 추가 옵션을 사용하는 환경에서는 위 명령에도 --extra qat 등을 지정합니다.
```

증류 노트북의 GKD 클래스는 최신 TRL의 `trl.experimental.gkd` 경로를 사용합니다.
`uv sync --upgrade`로 패키지 조합을 바꾸면 GPU 실행을 다시 검증해야 합니다.

## 검증 범위

2026-10-27에 로컬 Python 3.12 환경을 `uv sync --locked`로 설치했습니다.
노트북 8개의 형식 검사와 전체 코드 셀 164개의 Python/IPython 구문 검사를 통과했습니다.
각 노트북을 별도의 Runpod A40 Pod와 uv 환경에서 실행해 전체 164개 코드 셀의 실행을
완료했습니다. 원본 학습 스텝을 유지했고, Phone은 ExecuTorch 내보내기 후 실제 CPU에서
2토큰 순전파와 KV 캐시도 확인했습니다. 테스트 Pod 8개는 모두 중단했습니다.
노트북별 실행 조건, 패키지·코드 수정, 실행 증거와 검증 범위는
[A40 통합 실행 결과](../test_scenarios/unsloth/RESULTS.md)에 기록했습니다.

출처 및 패키지 관리 안내:

- [원본 Unsloth 노트북](https://github.com/unslothai/notebooks/tree/main/nb)
- [Unsloth 설치 안내](https://unsloth.ai/docs/get-started/install/pip-install)
- [uv 잠금 파일 및 동기화](https://docs.astral.sh/uv/concepts/projects/sync/)
