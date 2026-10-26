# Omni 모델 미세조정 요청과 결과

- 날짜: 2026-10-25 (Asia/Seoul)
- 범위: 마지막 학습 요청과 최종 결과 답변. 중간 진행 메시지는 생략했습니다.

## 사용자 요청

sft_t2a_mini, sft_a2a_mini, caspar/MiniMind-V-LLaVA-Short-Training-v1.0를 이용하여 omni model을 fine tuning 해. rtx 5090을 생성해서 학습하고 학습이 끝나면 결과를 보고하고 샘플 데이터를 보여줘. 학습된 모델은 huggingface에 업로드해. 학습이 완료되면 pod는 제거해

## 최종 답변

**학습·Hugging Face 업로드·Pod 삭제까지 완료했습니다.**

- RTX 5090 1개에서 5단계를 각각 1에포크 학습했습니다. 순수 학습 시간은 **약 74분**입니다.
- [학습된 모델 — 비공개 저장소](https://huggingface.co/caspar/MiniMind-O-Mini-LLaVA-5090-20261025)
- [전체 결과 보고서](../outputs/omni-mini-20261025/artifacts/REPORT_ko.md)
- 업로드 파일의 해시를 확인했고, Pod `zuq0hsnyvgsmjz`는 삭제했습니다.

검증 결과는 다음과 같습니다. 낮을수록 좋습니다.

| 평가 | 학습 전 → 최종 |
|---|---:|
| T2A 텍스트 손실 | 1.0167 → **0.9422** |
| T2A 오디오 손실 | 3.9385 → **3.9002** |
| A2A 텍스트 손실 | 1.2840 → **1.1678** |
| A2A 오디오 손실 | 3.4757 → **3.4468** |

이미지 QA 단계의 검증 손실은 **1.3586 → 1.0010**, 별도 테스트 690개의 손실은 **0.9808**입니다.

실제 생성 샘플입니다. **손실은 개선됐지만 의미 이해와 답변 정확성에는 오류가 남아 있습니다.**

| 입력 | 실제 모델 응답 | 확인 |
|---|---|---|
| `exuberant`의 덜 격식적인 동의어 | “expensive” | 오답 |
| 음성: 동물에서 얻는 제품 두 가지 | “dogs and cats” | 오답—참고 정답은 가죽·양모 |
| 이미지: 접시 위 조각 수 | “There are two slices…” | 수량 일치 |

[입력 이미지](../outputs/omni-mini-20261025/artifacts/samples/image-3-input.jpg) · [이미지 응답 음성](../outputs/omni-mini-20261025/artifacts/samples/image-3-output.wav) · [A2A 입력 음성](../outputs/omni-mini-20261025/artifacts/samples/a2a-1-input.wav) · [A2A 응답 음성](../outputs/omni-mini-20261025/artifacts/samples/a2a-1-output.wav)

보고서에 **전체 7개 샘플, 참고 정답, 생성 WAV, 자동 전사, 재현 설정**을 포함했습니다.
