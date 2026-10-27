# GPT 실습

Python 3.12와 uv를 사용하는 노트북 및 모델 실험 프로젝트입니다.
아래 명령은 `gpt/` 디렉토리에서 실행합니다.

```bash
uv sync --locked
uv run jupyter lab
```

VS Code에서는 `uv sync --locked` 실행 후 노트북 커널로
`gpt/.venv/bin/python`을 선택합니다.

모델 스크립트의 옵션은 다음과 같이 확인할 수 있습니다.

```bash
uv run python model_improvements/gpt_with_kv_gqa.py --help
```

노트북에서 사용하는 `previous_chapters.py`는
[heury/llm_hands_on 원본](https://github.com/heury/llm_hands_on/blob/main/previous_chapters.py)에서 가져왔습니다.
원본의 저작권 및 라이선스 안내를 파일 상단에 유지합니다.
데이터와 모델 가중치도 실습에 따라 별도로 내려받아야 합니다.

노트북별 A40 전체 실행 결과와 수정 사항은
[2026-10-27 검증 기록](../test_scenarios/gpt/README.md)을 참고하세요.
실제 실행 노트북·패키지 버전·로그와 재현 방법은
[validation/](validation/README.md)에 보관합니다.
