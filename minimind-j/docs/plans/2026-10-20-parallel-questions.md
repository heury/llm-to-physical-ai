# Parallel Typed Questions Implementation Plan

**Goal:** 공통 입력을 한 번 인코딩하고 choice / score / noul 질문을 같은 forward에서 판정한다.

**Architecture:** 공통 prefix + 독립 question branches를 packing한다. Block causal attention은 각 branch가 prefix와 자기 branch만 보게 하며 RoPE 위치를 branch마다 재설정한다. 별도 decision MLP를 제거하고 기존 LM head에서 A–Z 답변 코드 logits를 모아 질문별 CE를 계산한다. score는 bounded integer, noul은 boolean이다.

**Tech Stack:** Python, PyTorch, MiniMind, Transformers, pytest.

1. tests/test_decision.py: 한 backbone 호출, branch isolation, 질문 순서/배치 불변성, typed decoding, label 누출 방지, 학습/변환/복원 실패 테스트.
2. dataset.py: typed schema, legacy choice 정규화, packing, bool attention mask, local position ids, answer positions. max_length는 packed record 전체 길이.
3. model/model_minimind.py: bool 4D mask / batched positions 지원. 기존 2D causal 동작과 parameter names 유지.
4. model/decision.py: 기존 LM head, 후보 코드 logit gather, typed decode, freeze 시 tied LM head 분리, format v2 checkpoint 및 v1 재학습 오류.
5. train.py, decide.py, data/: 질문 단위 accuracy, JSON request CLI와 choice/score/noul 예제. 기존 단일 choice CLI 유지.
6. README.md, NOTICE, docs/verification.md: 구조/제한, CPU 실제 학습 및 독립 복사 검증. v1 검증 자료는 과거 기록으로 보존.

검증: pytest; CPU train → reload → JSON inference; 원래 MiniMind LM strict 변환/동결 검증; 독립 복사 pytest.
