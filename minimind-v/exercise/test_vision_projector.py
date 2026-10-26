# %% [1] 준비
# VS Code에서 셀을 위에서 아래로 실행하세요.
# 터미널: cd minimind-v && python exercise/test_vision_projector.py
from pathlib import Path
import sys

import torch

base = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
candidates = [base, base.parent, base / 'minimind-v']
project_dir = next((p for p in candidates if (p / 'model/model_vlm.py').is_file()), None)
if project_dir is None:
    raise RuntimeError('저장소 루트, minimind-v 또는 exercise에서 실행하세요.')
sys.path.insert(0, str(project_dir))
from model.model_vlm import MMVisionProjector

torch.manual_seed(42)


# %% [2] 샘플 데이터와 Projector 생성
# 이미지 1장, 패치 2개, 패치당 특징 4개: [B, P, D] = [1, 2, 4]
# 실제 이미지 인코더 출력 대신 직접 만든 작은 특징 벡터를 사용합니다.
x = torch.tensor([[[1., 2., 3., 4.],
                   [4., 1., 3., 2.]]])
projector = MMVisionProjector(in_dim=4, out_dim=3, source_tokens=2, target_tokens=1)
print('입력:', x)
print('구조:', projector)


# %% [3] 각 레이어의 출력 확인
# LayerNorm(4) → Linear(4,3) → GELU → Linear(3,3)
with torch.no_grad():
    h = x
    for layer in projector.mlp:
        h = layer(h)
        print(f'{type(layer).__name__}: shape={tuple(h.shape)}\n{h}')


# %% [4] forward 결과 검증
y = projector(x)
print('최종 출력:', y)
assert y.shape == (1, 2, 3)
assert torch.isfinite(y).all()
torch.testing.assert_close(y, h)
# 현재 구현은 특징 차원만 바꿉니다. target_tokens=1이어도 패치 수는 2입니다.
assert y.shape[1] == x.shape[1]
print('PASS: 특징 차원 4 → 3, 패치 수 2 유지')


# %% [5] 역전파와 가중치 업데이트 확인
# 학습된 정답이 아닌, 동작 확인을 위한 임의의 목표 벡터입니다.
target = torch.zeros(1, 2, 3)
optimizer = torch.optim.SGD(projector.parameters(), lr=0.01)
before = projector.mlp[1].weight.detach().clone()
optimizer.zero_grad()
loss = torch.nn.functional.mse_loss(projector(x), target)
loss.backward()
assert torch.isfinite(loss)
for name, parameter in projector.named_parameters():
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    print(f'{name}: gradient norm={parameter.grad.norm().item():.6f}')
optimizer.step()
assert not torch.equal(before, projector.mlp[1].weight)
print(f'PASS: loss={loss.item():.6f}, 가중치 업데이트 완료')
