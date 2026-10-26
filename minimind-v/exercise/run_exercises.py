# %% [1] 공통 환경 준비
"""작은 실제 모델로 MiniMindVLM의 데이터 흐름을 확인하는 CPU 실습.

실행: cd minimind-v && python exercise/run_exercises.py --case all
학습된 가중치나 인터넷은 필요하지 않습니다. assert 실패는 실습 실패입니다.
"""
import argparse
import tempfile
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

# 다른 프로젝트의 동명 model 패키지를 가져오지 않도록 경로를 고정합니다.
base = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
candidates = [base, base.parent, base / 'minimind-v']
project_dir = next((p for p in candidates if (p / 'model/model_vlm.py').is_file()), None)
if project_dir is None:
    raise RuntimeError('작업 디렉토리를 저장소 루트, minimind-v 또는 exercise로 설정하세요.')
sys.path.insert(0, str(project_dir))
from model.model_vlm import MiniMindVLM, VLMConfig, MMVisionProjector


try:
    _INTERACTIVE = get_ipython() is not None
except NameError:
    _INTERACTIVE = False

torch.set_num_threads(2)

# %% [2] 샘플 이미지 생성 함수
# [1]~[4]를 먼저 실행한 뒤 [5]~[11]은 원하는 순서로 실행하세요.
def make_images():
    """데이터 파일 대신 메모리에서 빨간 사각형/파란 원을 만듭니다."""
    square = Image.new('RGBA', (32, 32), 'white')
    ImageDraw.Draw(square).rectangle((4, 4, 27, 27), fill='red')
    circle = Image.new('RGB', (32, 32), 'white')
    ImageDraw.Draw(circle).ellipse((4, 4, 27, 27), fill='blue')
    return square, circle


# %% [3] 작은 모델 생성 함수
def make_model(directory):
    """다운로드 없이 작은 SigLIP을 저장하고 실제 get_vision_model로 읽습니다."""
    vision = SiglipVisionModel(SiglipVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, image_size=32, patch_size=16,
    ))
    vision.save_pretrained(directory)
    SiglipImageProcessor(size={'height': 32, 'width': 32}).save_pretrained(directory)
    config = VLMConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        max_position_embeddings=128, image_hidden_size=16, image_token_len=4,
        image_ids=[12], dropout=0.0, flash_attn=False,
    )
    return MiniMindVLM(config, vision_model_path=directory).eval()


# %% [4] 입력 데이터·독립 실행 준비
def sample(model):
    # 교육용 가상 토큰 ID: BOS=1, image=12, 질문=20, 답변=21, EOS=2.
    # 32x32 이미지 / 16x16 패치 → 4개 시각 토큰 → marker도 4개.
    ids = torch.tensor([[1, 12, 12, 12, 12, 20, 21, 2]])
    pixels = model.image2tensor(make_images()[0], model.processor)
    return ids, pixels


def run_exercise(exercise):
    """각 블록을 같은 초기 모델로 실행해 이전 학습의 영향을 제거합니다."""
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(prefix='minimind-v-exercise-') as directory:
        model = make_model(directory)
        exercise(model)
    print('PASS')


# %% [5] 이미지 전처리
def exercise_01(model):
    """설정, 로컬 인코더 로딩, RGBA→RGB, 비전 인코더 동결."""
    assert model.config.image_token_len == 4
    assert not model.vision_encoder.training
    assert all(not p.requires_grad for p in model.vision_encoder.parameters())
    assert model.get_vision_model('/__missing_exercise_vision__') == (None, None)
    _, pixels = sample(model)
    assert pixels['pixel_values'].shape == (1, 3, 32, 32)
    assert torch.isfinite(pixels['pixel_values']).all()
    print('RGBA 이미지 → pixel_values:', tuple(pixels['pixel_values'].shape))


if _INTERACTIVE:
    run_exercise(exercise_01)


# %% [6] 시각 특징 추출·투영
def exercise_02(model):
    """시각 특징 추출 → MLP 차원 변환. 토큰 개수는 변하지 않습니다."""
    _, pixels = sample(model)
    features = model.get_image_embeddings(pixels, model.vision_encoder)
    assert features.shape == (1, 4, 16) and not features.requires_grad
    projected = model.vision_proj(features)
    assert projected.shape == (1, 4, 32)
    # source_tokens/target_tokens 인자는 현재 projector에서 리샘플링하지 않습니다.
    projector = MMVisionProjector(16, 32, source_tokens=4, target_tokens=2)
    assert projector(features).shape[1] == 4
    print('특징 [B,P,Dv] → 투영 [B,P,Dlm]:', tuple(features.shape), '→', tuple(projected.shape))


if _INTERACTIVE:
    run_exercise(exercise_02)


# %% [7] 이미지 토큰 교체
def exercise_03(model):
    """연속 marker 구간을 이미지 특징으로 교체; 텍스트 위치는 보존."""
    ids = torch.tensor([[1, 12, 12, 12, 12, 20, 12, 12, 12, 12, 2]])
    hidden = model.model.embed_tokens(ids)
    # 첫 이미지는 1, 두 번째 이미지는 2: 교체 결과를 눈으로 이해하기 쉽습니다.
    features = torch.stack([torch.ones(4, 32), torch.full((4, 32), 2.)])[None]
    merged = model.count_vision_proj(ids, hidden, features, seqlen=ids.shape[1])
    torch.testing.assert_close(merged[:, 1:5], features[:, 0])
    torch.testing.assert_close(merged[:, 6:10], features[:, 1])
    torch.testing.assert_close(merged[:, [0, 5, 10]], hidden[:, [0, 5, 10]])
    assert model.count_vision_proj(ids, hidden, None) is hidden
    single = model.count_vision_proj(ids[:, :6], hidden[:, :6], features[:, 0], seqlen=6)
    torch.testing.assert_close(single[:, 1:5], features[:, 0])
    print('두 이미지 marker 구간 교체 성공; 텍스트 임베딩 보존')


if _INTERACTIVE:
    run_exercise(exercise_03)


# %% [8] 단일·다중 이미지 forward
@torch.no_grad()
def exercise_04(model):
    """텍스트만/단일 이미지/다중 이미지 forward와 출력 선택."""
    ids, pixels = sample(model)
    text = model(ids)
    single = model(ids, pixel_values=pixels)
    assert single.logits.shape == (1, 8, 64)
    assert torch.isfinite(single.logits).all()
    assert not torch.allclose(text.logits, single.logits)
    torch.testing.assert_close(model(ids, pixel_values=pixels, logits_to_keep=1).logits,
                               single.logits[:, -1:])
    indices = torch.tensor([0, 7])
    torch.testing.assert_close(model(ids, pixel_values=pixels, logits_to_keep=indices).logits,
                               single.logits[:, indices])
    # 복수 이미지: dict 내부 [B,N,C,H,W]. 이미지 사이에는 텍스트 토큰이 필요합니다.
    pair = torch.cat([model.image2tensor(im, model.processor)['pixel_values']
                      for im in make_images()], dim=0)[None]
    multi_ids = torch.tensor([[1, 12, 12, 12, 12, 20, 12, 12, 12, 12, 21]])
    multi = model(multi_ids, pixel_values={'pixel_values': pair})
    assert multi.logits.shape == (1, 11, 64) and torch.isfinite(multi.logits).all()
    print('단일 이미지 logits:', tuple(single.logits.shape), '/ 다중 이미지:', tuple(multi.logits.shape))


if _INTERACTIVE:
    run_exercise(exercise_04)


# %% [9] 손실 계산·역전파
def exercise_05(model):
    """답변만 loss 계산, projector만 한 번 업데이트, 인코더는 고정."""
    ids, pixels = sample(model)
    labels = ids.clone()
    labels[:, :6] = -100  # 질문/이미지는 정답에서 제외, 답변21과 EOS2만 학습.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.vision_proj.parameters():
        parameter.requires_grad_(True)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.vision_proj.parameters(), lr=1e-3)
    optimizer.zero_grad()
    output = model(ids, pixel_values=pixels, labels=labels)
    expected = F.cross_entropy(output.logits[:, :-1].reshape(-1, 64), labels[:, 1:].reshape(-1))
    torch.testing.assert_close(output.loss, expected)
    (output.loss + output.aux_loss).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.vision_proj.parameters())
    assert all(p.grad is None for p in model.vision_encoder.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    changed = [name for name, p in model.named_parameters() if not torch.equal(before[name], p)]
    assert changed and all(name.startswith('vision_proj.') for name in changed)
    print(f'답변 토큰 2개 loss={output.loss.item():.4f}; projector만 업데이트: {len(changed)}개 매개변수')


if _INTERACTIVE:
    run_exercise(exercise_05)


# %% [10] KV 캐시
@torch.no_grad()
def exercise_06(model):
    """KV 캐시가 있는 다음 토큰에서는 이미지 인코더를 다시 실행하지 않습니다."""
    ids, pixels = sample(model)
    calls = []
    handle = model.vision_encoder.register_forward_hook(lambda *args: calls.append(1))
    try:
        prefix = model(ids[:, :-1], pixel_values=pixels, use_cache=True)
        cached = model(ids[:, -1:], pixel_values=pixels, past_key_values=prefix.past_key_values,
                       use_cache=True)
        assert len(calls) == 1
        assert cached.past_key_values[0][0].shape[1] == ids.shape[1]
        full = model(ids, pixel_values=pixels)
        torch.testing.assert_close(cached.logits, full.logits[:, -1:], atol=1e-5, rtol=1e-4)
    finally:
        handle.remove()
    print('캐시 사용/전체 재계산 logits 일치; 후속 토큰의 이미지 재인코딩 없음')


if _INTERACTIVE:
    run_exercise(exercise_06)


# %% [11] 토큰 생성
@torch.no_grad()
def exercise_07(model):
    """이미지 조건 생성, 여러 반환 시퀀스, 캐시 유무 비교."""
    ids, pixels = sample(model)
    prompt = ids[:, :6]
    options = dict(pixel_values=pixels, max_new_tokens=4, eos_token_id=None,
                   do_sample=False, temperature=1., top_k=0, top_p=1.)
    cached = model.generate(prompt, use_cache=True, **options)
    uncached = model.generate(prompt, use_cache=False, **options)
    torch.testing.assert_close(cached, uncached)
    multiple = model.generate(prompt, num_return_sequences=2, **options)
    assert multiple.shape == (2, 10)
    torch.testing.assert_close(multiple, cached.repeat(2, 1))
    torch.manual_seed(42)
    options.update(do_sample=True, temperature=.8, top_k=10, top_p=.9)
    sampled = model.generate(prompt, num_return_sequences=2, **options)
    assert sampled.shape == (2, 10)
    print('greedy 생성 ID:', cached[:, 6:].tolist())
    print('sample 생성 ID:', sampled[:, 6:].tolist(), '(무작위 모델: 문장 품질 평가 아님)')


if _INTERACTIVE:
    run_exercise(exercise_07)


# %% [12] 명령행 실행
# Interactive Window에서는 이 셀을 실행해도 argparse를 호출하지 않습니다.
CASES = {str(i): globals()[f'exercise_{i:02d}'] for i in range(1, 8)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', *CASES], default='all')
    args = parser.parse_args()
    torch.set_num_threads(2)
    selected = CASES if args.case == 'all' else {args.case: CASES[args.case]}
    with tempfile.TemporaryDirectory(prefix='minimind-v-exercise-') as directory:
        for number, exercise in selected.items():
            # 각 실습은 동일한 초기 상태로 독립 실행 (학습 실습의 영향 방지).
            torch.manual_seed(42)
            model = make_model(directory)
            print(f'\n[{number}] {exercise.__doc__}')
            exercise(model)
            print('PASS')
    print(f'\n완료: {len(selected)}개 실습 통과')


if __name__ == '__main__' and not _INTERACTIVE:
    main()
