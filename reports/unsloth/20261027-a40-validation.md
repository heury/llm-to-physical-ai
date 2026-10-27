# Unsloth 노트북 A40 전체 실행 테스트

요청: 노트북마다 A40 Pod 하나를 사용해 전체 코드 셀을 실행하고 오류와 소요 시간을 확인한 뒤 테스트 Pod를 제거한다.

전체 8개 중 8개 통과, 0개 실패. 완료한 코드 셀 164/164개. 이번 테스트 Pod 8개의 제거를 각각 확인했다.

## 실행 조건

- 실행 날짜: 2026-10-27 (Asia/Seoul). 각 노트북 원본의 학습 스텝과 셀을 유지했다.
- GPU: NVIDIA A40 48GB, 1 GPU/Pod, Secure Cloud. 카탈로그 조회 기록: `gpu-catalog.json`.
- 이미지: `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster` (공식 템플릿 카탈로그에서 확인), 최소 CUDA 13.0.
- Pod마다 40GB 컨테이너 디스크와 150GB Pod 볼륨을 사용했다. 테스트 증거를 로컬로 회수한 뒤 Pod와 해당 Pod 디스크를 제거했다.
- GPU 카탈로그 요금: Pod당 $0.59/시간. 실제 생성 응답의 `costPerHr`도 $0.59. 디스크·부가 요금은 포함하지 않은 GPU 기준 금액이며 청구서 조회 금액이 아니다.
- 독립 uv 환경에서 `uv sync --locked --reinstall-package xformers`를 실행하고 노트북에 필요한 extra만 지정했다. `uv pip check` 후 저장소의 `scripts/execute_notebook.py`로 실행했다.
- 표의 시간에는 Pod 예약·부팅, 소스 전송, 결과 회수와 Pod 제거에 걸린 시간을 포함하지 않는다. 설치 시간은 패키지 다운로드·설치를 포함한다. 노트북 시간은 커널 시작, 모델·데이터 다운로드, 최초 컴파일, 학습·추론·저장을 포함한 벽시계 시간이다. 순수 학습 시간 또는 예열 후 성능으로 해석하지 않는다.
- 모델·데이터는 원본 노트북의 Hugging Face ID와 다운로드 설정을 그대로 사용했다. revision이 명시되지 않은 항목은 실행 시점의 기본 브랜치를 사용하므로 소스·패키지 잠금만으로 모델·데이터 바이트까지 고정한 재현성은 보장하지 않는다. 회수하지 못한 리비전은 미확인으로 남긴다.
- 비활성화된 선택적 셀(`if False`, 주석)은 실행 범위에 포함되지 않는다. Hugging Face 업로드는 수행하지 않았다.
- 소스 커밋: `bdd759b65aaefa800a3bbb0584404598b1208373`. 노트북 SHA256과 코드 셀 수는 `source-manifest.json`에 기록했다.

## 결과

| 노트북 | Pod ID | extra | 환경 설치 | 노트북 실행 | 완료한 코드 셀 | 결과 | Pod 제거 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Qwen3_(0.6B)-Knowledge_Distillation.ipynb | `xiljg73n649fmv` | 기본 | 3분 33.3초 | 4분 30.3초 | 16/16 | passed | 제거 확인 |
| Qwen3_(0_6B)-Phone_Deployment.ipynb | `sepokaxt49kfsw` | phone | 3분 13.6초 | 22분 31.7초 | 22/22 | passed | 제거 확인 |
| Qwen3_(4B)-GRPO.ipynb | `bsdvflk17nixxo` | grpo | 4분 47.2초 | 76분 31.9초 | 35/35 | passed | 제거 확인 |
| Qwen3_(4B)-Tool_Calling.ipynb | `rrwbxlzqjlsu57` | tools | 3분 29.1초 | 8분 52.3초 | 17/17 | passed | 제거 확인 |
| Qwen3_(4B)_Instruct-QAT.ipynb | `1s916rao1r318p` | qat | 3분 53.9초 | 9분 25.4초 | 22/22 | passed | 제거 확인 |
| Qwen3_5_(4B)-Decision.ipynb | `bb3q60fcviamxq` | 기본 | 2분 53.7초 | 17분 54.7초 | 15/15 | passed | 제거 확인 |
| Qwen3_5_(4B)_Vision.ipynb | `bsxw8bqu4g08c7` | 기본 | 2분 36.0초 | 8분 51.0초 | 20/20 | passed | 제거 확인 |
| Qwen3_5_(4B)_Vision_GRPO.ipynb | `3zd9gxkz4z1xqu` | 기본 | 2분 36.8초 | 91분 32.3초 | 17/17 | passed | 제거 확인 |

## 주요 구간별 시간

아래 값은 해당 코드 셀의 벽시계 시간으로, 최초 컴파일·학습 초기화·체크포인트 저장 등이 포함될 수 있다. 정확한 셀별 시간은 각 작업 폴더의 `cell-timings.csv`에 기록했다.

| 작업 | 셀 번호 (0부터) | 구간 | 소요 시간 |
| --- | ---: | --- | ---: |
| kd | 33 | 학습 | 0분 38.4초 |
| phone | 28 | 학습 | 3분 50.2초 |
| phone | 36 | ExecuTorch 내보내기 | 12분 42.8초 |
| grpo | 21 | 학습 | 0분 53.9초 |
| grpo | 55 | 학습 | 60분 28.0초 |
| tools | 20 | 학습 | 4분 12.2초 |
| qat | 27 | 학습 | 2분 43.5초 |
| decision | 14 | 학습 | 12분 6.5초 |
| vision | 23 | 학습 | 3분 6.4초 |
| vision-grpo | 24 | 학습 | 82분 41.2초 |

## 오류와 검증 범위

실행 중 종료를 유발한 오류는 없었다. 모든 환경에서 의존성 검사를 통과했으며, 각 노트북의 활성 코드 셀이 끝까지 완료됐다. 이는 실행 가능성 확인이며 생성 답변의 정확성이나 학습 성능 향상을 보증하는 평가는 아니다. Phone의 `.pte` 생성과 CPU 순전파·KV 캐시는 노트북 내 검증까지 통과했다. 선택적으로 비활성화된 업로드·내보내기 경로는 검증하지 않았다.

저장된 학습 표의 손실 434개를 확인했으며 NaN·무한대 값은 없었다. 검사 내역은 `training-metrics-check.json`에 저장했다. 원본 노트북 8개의 SHA256이 실행 전과 동일한 것도 확인했다.

## 실행 중 확인한 경고

- Qwen3.5 계열에서 `causal_conv1d` 미설치에 따른 PyTorch 대체 경로 경고가 발생했다. 측정 시간은 저장소의 잠금 환경 그대로 실행한 결과이며 해당 최적화 커널을 추가한 성능은 검증하지 않았다.
- Hugging Face 비인증 다운로드 경고와 일부 PyTorch/vLLM API 사용 중단 예정 경고가 나타났다. 경고 자체를 실패로 처리하지 않고 실제 셀 종료 상태를 판정에 사용했다.

## 증거와 재현

- 실행 증거 폴더: [outputs/unsloth/20261027-a40-validation](../../outputs/unsloth/20261027-a40-validation/). `outputs/`는 Git 추적에서 제외되므로 이 보고서와 별도로 보관해야 한다.
- 노트북별 `<작업명>/evidence/`에 실행 사본(`executed.ipynb`), 셀별 상태·패키지 버전(`status.json`), 단계별 시간(`timing.json`), 로그와 생성 파일 목록(`artifact-manifest.json`)을 보관한다. 대용량 모델 가중치와 내보낸 `.pte` 바이너리는 회수 대상에서 제외하고 크기와 SHA256을 목록에 기록했다.
- `<작업명>/deleted.json`은 제거 후 Pod 목록에서 ID가 사라진 것을 확인한 기록이다. 기존 사용자 Pod는 변경하지 않는다.
- 테스트 절차: 동일 이미지와 GPU에서 소스 노트북·pyproject.toml·uv.lock을 복사하고 해당 extra로 uv 환경을 설치한다. `uv run --no-sync python scripts/execute_notebook.py <노트북> --output-dir <결과 경로>`를 실행한다.

최종 `runpodctl pod list` 재조회에서도 테스트 Pod ID 8개가 모두 없는 것을 확인했다. 기록은 `pods-after.json`과 `cleanup-verification.json`에 저장했다.
