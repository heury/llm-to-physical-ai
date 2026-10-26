# TalkerEmbedding, TalkerHead, TalkerModule 이해하기

세 모듈은 **이전에 생성한 음성을 읽고 → 다음 음성을 계산하고 → 다음 음성 코드를 예측하는 과정**을 담당합니다.

여기서 음성 코드는 파형 자체가 아니라 Mimi 코덱이 음성을 표현하는 정수 토큰입니다. MiniMind-O는 8개 코드북의 토큰을 조합해 음성을 표현합니다. 코드북은 음성을 표현하는 코드들의 사전이라고 생각하면 됩니다.

## 전체 흐름

```text
이전 음성 코드 8개                  Thinker의 중간층 표현
        │                              (말할 내용)
        ▼                                  │
 TalkerEmbedding                            │
 “코드를 벡터로 읽기”                       │
        │                                  │
    codec_proj                         embed_proj
        │                                  │
        └──── 배율을 곱한 뒤 더하기 ────────┘
                         │
                         ▼
              Talker Transformer + norm
              “문맥을 보고 다음 출력 계산”
                         │
                         ▼
                     TalkerHead
              “코드북별 다음 토큰 점수 계산”
                         │
                         ▼
              토큰 선택 및 지연 스케줄 처리
                         │
                         ▼
                Mimi 디코더 → 음성 파형
```

`TalkerModule`은 이 과정에 필요한 임베딩, 변환층, Transformer, 출력 헤드를 묶어 둔 모듈입니다. 토큰 선택과 Mimi 디코딩은 별도의 생성 경로에서 처리합니다.

## 1. TalkerEmbedding: 음성 코드를 읽는 입구

텍스트 모델이 단어 ID를 임베딩 벡터로 바꾸듯, `TalkerEmbedding`은 **8개 코드북의 정수 토큰을 모델이 읽을 수 있는 벡터로 바꿉니다.**

각 코드북에 대해 공통 임베딩과 전용 어댑터의 출력을 더합니다.

```text
코드북 i의 표현
  = 공유 임베딩(토큰 i) + 코드북 i 전용 어댑터(토큰 i)

최종 음성 표현 = 8개 코드북 표현의 평균
```

공유 임베딩은 공통 부분을 담당합니다. 전용 어댑터는 같은 토큰 번호라도 어느 코드북에 속했는지에 따라 다른 표현을 만들 수 있게 합니다.

기본 설정의 내부 구조는 다음과 같습니다.

| 구성 | 연산 |
|---|---|
| 공유 `base` | Embedding(2112, 768) |
| 코드북별 `adapter` × 8 | Embedding(2112, 256) → GELU → Linear(256 → 768) |

입력과 출력의 형태는 다음과 같습니다. `B`는 배치 크기, `T`는 시퀀스 길이입니다.

```text
입력: [B, 8, T]    — 코드북별 정수 토큰
출력: [B, T, 768]  — 8개 표현을 평균한 벡터
```

즉, 코드북별 벡터를 길게 이어 붙이는 방식이 아니라 **평균으로 하나의 벡터에 합치는 방식**입니다.

## 2. TalkerHead: 다음 음성 코드를 예측하는 출구

`TalkerHead`는 **Talker가 계산한 벡터를 받아 코드북별 다음 토큰 점수(logits)를 출력**합니다.

핵심 구현은 다음과 같습니다.

```python
base_out = self.base(x)
return [base_out + adapter(x) for adapter in self.adapters]
```

각 코드북의 점수는 공유 출력층과 전용 어댑터의 합으로 만들어집니다.

```text
코드북 i의 점수
  = 공유 출력층(x) + 코드북 i 전용 어댑터(x)
```

기본 설정의 구조는 다음과 같습니다.

| 구성 | 연산 | 역할 |
|---|---|---|
| 공유 `base` | Linear(768 → 2112) | 모든 코드북의 공통 점수 |
| 코드북별 `adapter` × 8 | Linear(768 → 256) → GELU → Linear(256 → 2112) | 코드북별 보정값 |

```text
입력: [B, T, 768]
출력: [B, T, 2112] 텐서 8개를 담은 리스트
```

`2112`는 특수 토큰 등을 포함하는 설정상의 오디오 어휘 크기입니다. 예를 들어 코드북 0에서는 토큰 125의 점수가 높고, 코드북 1에서는 토큰 830의 점수가 높을 수 있습니다. 이후 생성 로직이 점수를 바탕으로 토큰을 선택합니다. Head 자체는 softmax나 샘플링을 하지 않습니다.

### 공유 출력층을 사용하는 이유

코드북마다 큰 출력층 전체를 따로 두는 대신, 공통 부분을 공유하고 작은 중간 차원인 256을 거치는 어댑터를 사용하면 매개변수를 줄일 수 있습니다.

기본 설정에서 bias 없는 독립 출력층 8개와 비교하면 다음과 같습니다.

| 방식 | 매개변수 수 |
|---|---:|
| 독립적인 Linear(768 → 2112) × 8 | 12,976,128 |
| 현재의 공유 출력층 + 어댑터 8개 | 7,520,256 |

출력 헤드의 매개변수가 약 42% 줄어듭니다. 이는 Talker 전체가 아니라 `TalkerHead`만 비교한 수치입니다.

작은 차원을 거치는 어댑터라는 점은 LoRA를 연상시키지만, 이 구현에는 GELU가 있고 공유 출력층도 함께 학습하므로 일반적인 LoRA와는 다릅니다.

## 3. TalkerModule: 음성 생성 부품 전체

`TalkerModule`에는 앞의 두 모듈과 문맥을 계산하는 Transformer 등이 들어 있습니다.

| 부품 | 역할 |
|---|---|
| `embed_tokens` — TalkerEmbedding | 이전 음성 코드를 벡터로 변환 |
| `embed_proj` | Thinker 중간층 표현을 Talker 차원으로 변환 |
| `codec_proj` | 음성 임베딩을 Talker 입력에 맞게 가공 |
| `spk_proj` | 화자 임베딩을 Talker 차원으로 변환 |
| `text_scale`, `audio_scale` | 두 입력에 곱하는 학습 가능한 배율 |
| `layers` | 문맥을 처리하는 Transformer 블록, 기본 4층 |
| `norm` | Transformer 출력에 RMSNorm 적용 |
| `lm_head` — TalkerHead | 코드북별 다음 토큰 점수 계산 |
| `freqs_cos`, `freqs_sin` | RoPE 위치 정보 계산에 사용하는 버퍼 |

Talker 입력은 다음과 같이 만들어집니다.

```text
Talker 입력
  = embed_proj(Thinker 중간층 표현) × text_scale
  + codec_proj(음성 임베딩) × audio_scale
```

두 배율의 초기값은 각각 `3.0`, `1.0`이며 학습 중 바뀔 수 있습니다. 이 결합을 통해 Talker는 **말할 내용과 지금까지 생성한 음성 흐름을 함께 참고**합니다. `text_scale`이라는 이름을 사용하지만 실제 입력은 텍스트 토큰 자체가 아니라 Thinker의 중간층 은닉 상태입니다.

화자 임베딩이 주어지면 `spk_proj`로 변환한 값을 지정된 화자 토큰 위치의 음성 임베딩 대신 넣습니다.

구현상 `TalkerModule` 자체에는 `forward`가 정의되어 있지 않습니다. 부품은 `TalkerModule`에서 만들고, 실제 연결과 실행은 `MiniMindOmni.forward` 안에서 수행합니다.

## 헷갈리기 쉬운 점

- `TalkerHead`와 `TalkerEmbedding`의 `num_layers=8`은 코드북별 분기 수입니다. Talker Transformer의 깊이는 `num_talker_hidden_layers`로 따로 지정하며 기본 4층입니다.
- Head의 8개 분기는 같은 은닉 상태를 받아 함께 점수를 계산합니다. 앞 헤드가 선택한 토큰을 다음 헤드에 차례대로 넣는 구조는 아닙니다.
- 생성에는 코드북별 지연 스케줄이 적용됩니다. 따라서 한 생성 스텝의 출력 8개가 모두 같은 음성 프레임을 가리킨다고 해석하면 안 됩니다.
- Embedding과 Head는 각각 입력과 출력을 담당하지만, 서로를 정확히 되돌리는 역함수는 아닙니다.
- 여기의 차원과 개수는 현재 기본 설정 기준입니다. 특히 코드북 수 8은 생성 경로에도 사용되므로, Head의 분기 수만 바꿔서 전체 코드북 수를 변경할 수는 없습니다.

## 관련 코드

- [model/model_omni.py](../model/model_omni.py): `OmniConfig`, `TalkerEmbedding`, `TalkerHead`, `TalkerModule`, `MiniMindOmni.forward`와 생성 경로
- [README_ko.md](../README_ko.md): Thinker–Talker 구조, 중간층 Bridge, MTP와 스트리밍 설명

**Embedding은 입력 변환, Head는 출력 예측, Module은 이들을 포함한 전체 음성 생성 구성**으로 기억하면 됩니다.
