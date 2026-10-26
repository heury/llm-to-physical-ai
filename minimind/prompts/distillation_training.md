# Soft knowledge distillation 학습

기록일: 2026-10-25 (KST)

사용자 요청과 주요 답변을 대화 순서대로 보존하고, 중간 실행 과정은 별도로 요약했다. 도구 호출 원문과 환경 설정 메시지는 제외했다.

## user — 최초 요청

Qwen/Qwen3-1.7B를 teacher로,  caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0를 student로 해서 soft knowledge distillation 학습을 진행해. 학습 데이터는 caspar/MiniMind-Mixed-SFT-Training-v1.0에서 gsm8k를 이용해.  학습이 끝나면 학습 결과와 caspar/MiniMind-Mixed-SFT-Training-v1.0와 학습된 모델을 gsm8k의 테스트셋으로 평가해. 학습 모델은 huggingface에 업로드해

## agent — 최초 요청 확인 및 조사 요약

비교 대상으로 적힌 데이터셋 이름은 원본 student 모델 `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`를 뜻하는 것으로 해석했다.

기존 증류 코드는 teacher와 student가 같은 토크나이저를 사용한다고 가정한다. Qwen과 MiniMind는 어휘가 달라 기존 코드처럼 teacher logits를 student 어휘 크기로 잘라 비교하면 올바른 증류가 되지 않는다. 서로 다른 토크나이저를 지원하는 정렬·변환 방식이 필요하다.

로컬 GPU 드라이버를 사용할 수 없어 원격 실행 환경을 조사했다. Hugging Face 계정과 요청한 모델·데이터셋 접근을 확인했으나, Qwen teacher 학습은 시작하지 않았다. 다음 질문으로 대화가 전환되었다.

## user — Unsloth 예제 질문

[https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_(0.6B)-Knowledge_Distillation.ipynb](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_%280.6B%29-Knowledge_Distillation.ipynb)
을 보면 서로 다른 모델끼리 distillation을 하고 있는데 soft distillation이야?

## agent

네, **해당 노트북은 teacher의 출력 확률분포를 student가 따라 배우는 soft distillation입니다.**

코드를 보면 다음과 같습니다.

- **Teacher:** `Qwen3-1.7B`
- **Student:** `Qwen3-0.6B`
- **학습:** `GKDTrainer`로 두 모델의 출력 분포 차이를 줄입니다.
- **설정:** `lmbda=0.0`은 데이터셋의 응답을 사용하는 off-policy 학습이고, `beta=0.5`는 generalized Jensen–Shannon divergence 설정입니다. `temperature=1.0`이어도 확률분포로 학습하므로 soft distillation입니다.

핵심은 **모델이 달라도 되지만, 토큰별 확률을 직접 비교하려면 토큰 ID의 의미가 같아야 한다**는 점입니다. 노트북에도 다음 조건이 명시되어 있고, 어휘 크기와 토큰화 결과를 검사합니다.

> “student and teacher must share a vocabulary”

[노트북 원문](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_%280.6B%29-Knowledge_Distillation.ipynb)

따라서 두 경우는 이렇게 다릅니다.

| 조합 | 노트북의 직접 분포 비교 |
|---|---|
| Qwen3-1.7B → Qwen3-0.6B | 공유 어휘를 사용하므로 가능 |
| Qwen3-1.7B → MiniMind-64M | 토크나이저와 어휘가 달라 그대로 적용 불가 |

앞서 말씀드린 것도 **서로 다른 모델이라서 불가능하다는 뜻은 아니었습니다.** 요청하신 Qwen → MiniMind 역시 soft distillation이 가능하지만, 토큰 정렬이나 공통 확률 공간으로의 변환을 지원하는 별도 방식이 필요합니다. 기존 코드처럼 Qwen logits를 MiniMind 어휘 크기로 잘라 비교하면 올바른 증류가 되지 않습니다.

## user — Teacher 변경 및 실행 요청

MiniMind-MoE-198M-A64M-Mixed-SFT-v1.0를 teacher로,  caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0를 student로 해서 soft knowledge distillation 학습을 진행해. 학습 데이터는 caspar/MiniMind-Mixed-SFT-Training-v1.0에서 gsm8k를 이용해.  학습이 끝나면 학습 결과와 caspar/MiniMind-Mixed-SFT-Training-v1.0와 학습된 모델을 gsm8k의 테스트셋으로 평가해. 학습 모델은 huggingface에 업로드해

## agent — 진행 과정 요약

Teacher를 `caspar/MiniMind-MoE-198M-A64M-Mixed-SFT-v1.0`로 변경했다. 비교 대상은 이전과 같이 원본 Dense student로 해석했다.

- 두 모델의 토크나이저 파일과 6,400개 어휘가 일치함을 확인했다.
- 공개 데이터셋의 GSM8K 학습 대화 6,851개만 사용했다. 전처리 결과는 7,333개 윈도우, epoch당 assistant 정답 토큰 845,130개로 공개 manifest와 일치했다.
- 340토큰 윈도우, stride 256을 사용했다. Prompt·padding·중복 overlap 타깃을 마스킹하고 마지막 짧은 윈도우와 EOS 뒤 줄바꿈 타깃을 보존했다.
- Teacher는 고정하고 student 전체 매개변수를 3 epoch, 345 updates 학습했다.
- 손실은 `0.5 × CE + 0.5 × T² × KL(teacher || student)`, 온도는 `T=1.5`였다. 전체 어휘 확률분포를 비교했다.
- Microbatch 16, accumulation 4, cosine 학습률 `5e-6 → 5e-7`, AdamW weight decay 0.01, gradient clipping 1, seed 42를 사용했다. 실제 타깃 토큰 수로 손실을 정규화했다.
- A40 GPU에서 BF16 연산과 FP32 학습 가중치를 사용했다. 당시 확인한 GPU 요금은 시간당 $0.59였다.
- 초기 데이터 검증에서 체크섬 객체 형식과 EOS 뒤 줄바꿈 처리 차이를 수정했다. 이 단계에서는 optimizer update가 실행되지 않았다. 중지한 최초 Pod의 호스트에 GPU 여유가 없어 해당 Pod를 삭제하고 동일 볼륨을 연결한 새 A40 Pod에서 진행했다.
- 손실 shifting/masking, 동일 분포 KL, 정답 추출 단위 테스트 3개와 실제 GPU backward·teacher 고정 검사가 통과했다.
- 공식 GSM8K main/test 1,319문항을 사용했다. 릴리스의 이름은 `validation_gsm8k`이며, 학습·하이퍼파라미터 조정·체크포인트 선택에는 사용하지 않았다.
- 최종 FP16 모델을 저장·재로딩하고 optimizer 체크포인트와 텐서가 일치하는지 검증한 뒤 평가했다. 원본 Dense도 고정된 원본 리비전에서 불러와 같은 조건으로 평가했다.
- 결과 파일 해시와 Hugging Face 게시 commit에서 다시 받은 가중치 해시를 검증했다. 두 작업 Pod는 삭제하고 기존 볼륨과 재개용 체크포인트는 보존했다.

## agent — 최종 답변

학습·평가·[Hugging Face 업로드](https://huggingface.co/caspar/MiniMind-Dense-64M-GSM8K-SoftDistill-MoE-v1.0)를 완료했습니다.

GSM8K 학습 대화 **6,851개**, **3 epoch·345 updates**로 MoE → Dense soft distillation을 진행했습니다.

테스트 1,319문항을 동일한 greedy·340토큰 문맥 조건으로 평가했습니다.

| 지표 | 원본 Dense | 증류 모델 |
|---|---:|---:|
| 엄격 정답률 | 13/1,319 **(0.99%)** | 12/1,319 **(0.91%)** |
| 마지막 숫자 허용 | 1.52% | 1.59% |
| CE | 1.0005 | 0.9333 |
| Teacher KL | 0.5934 | 0.4556 |

**손실은 감소했지만, 엄격 정답률 개선은 확인되지 않았습니다.**

게시 모델의 해시 검증과 작업 GPU 삭제도 완료했습니다. 설정·학습 로그·평가 결과는 [상세 보고서](../outputs/moe-dense-gsm8k-distill-20261025/REPORT.md)에 기록했습니다.

## 재현 및 결과 위치

- [학습·평가·게시 코드](../scripts/runpod_gsm8k_distill/README.md)
- [고정된 teacher/student/dataset 리비전](../scripts/runpod_gsm8k_distill/sources.json)
- [상세 결과 보고서](../outputs/moe-dense-gsm8k-distill-20261025/REPORT.md)
- [게시 검증 기록](../outputs/moe-dense-gsm8k-distill-20261025/published.json)
- 게시 commit: `a1dfda6d4df497c106327c41da96c6696e6a26af`
- 모델 SHA256: `fcd32854e2fa73e542a9fc4ce706fd00e0944475a8103f621feadc8f0b703056`
- 재개용 체크포인트: 기존 Runpod 볼륨의 `/workspace/moe-dense-gsm8k-distill-20261025/resume.pt`

## user — 대화 저장 요청

이 대화 prompts에 저장해
