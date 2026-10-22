"""교육용 VLA: 관측을 읽고, 노이즈 청크를 연속값 행동 청크로 변환한다.

학습: forward → encode_context → flow_loss → velocity
추론: sample_actions → encode_context → integrate_actions → velocity 반복

B: 배치, L: 문맥 토큰 수, H: 행동 청크 길이, A: 행동 차원.
생성 시간 t는 0=노이즈, 1=행동이며 로봇의 실제 실행 시간과 별개다.
SmolVLA의 개념을 단순화했으며 그 체크포인트와 호환되지 않는다.
"""

import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from .model_vlm import MiniMindVLM, VLMConfig


@dataclass
class VLAConfig:
    vlm_config: dict = field(
        default_factory=lambda: dict(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=192,
            vocab_size=6400,
            max_position_embeddings=256,
            image_hidden_size=768,
            image_token_len=64,
        )
    )
    vision_model_path: str = field(
        default_factory=lambda: str(
            Path(__file__).resolve().parent / "siglip2-base-p32-256-ve"
        )
    )
    state_dim: int = 2
    action_dim: int = 2
    chunk_size: int = 8
    expert_dim: int = 64
    expert_heads: int = 4
    expert_layers: int = 2
    freeze_vlm: bool = False

    def __post_init__(self):
        if self.vision_model_path == "tiny":
            raise ValueError(
                "Separate tiny vision encoder is no longer supported; provide SigLIP weights"
            )
        for name in (
            "state_dim",
            "action_dim",
            "chunk_size",
            "expert_dim",
            "expert_heads",
            "expert_layers",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.expert_dim % self.expert_heads or self.expert_dim % 2:
            raise ValueError("expert_dim must be even and divisible by expert_heads")
        if self.vlm_config.get("use_moe", False):
            raise ValueError("This teaching version supports dense MiniMind only")


class EducationalVLM(MiniMindVLM):
    """Native SigLIP + first half of MiniMind-V language layers, without logits."""

    def __init__(self, config, vision_model_path):
        config = deepcopy(config)
        if type(config.num_hidden_layers) is not int or config.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be a positive integer")
        # Construct only the retained prefix; do not allocate the unused tail.
        config.num_hidden_layers = max(1, config.num_hidden_layers // 2)
        super().__init__(config, vision_model_path)
        if self.vision_encoder is None or self.processor is None:
            raise ValueError(f"SigLIP model/processor not found: {vision_model_path}")

    def forward(self, input_ids, attention_mask, pixel_values):
        # Reuse native SigLIP/projector/language layers; skip vocabulary logits.
        image = self.get_image_embeddings(
            {"pixel_values": pixel_values}, self.vision_encoder
        )
        if image.shape[1:] != (
            self.config.image_token_len,
            self.config.image_hidden_size,
        ):
            raise ValueError("image encoder shape does not match VLM config")
        # encode_context가 검증한 단일 이미지 prefix를 한 번에 교체한다.
        # 원본 MiniMind-V처럼 텍스트 embedding에 dropout을 적용한 뒤 삽입한다.
        h = self.model.dropout(self.model.embed_tokens(input_ids))
        h = torch.cat(
            [self.vision_proj(image), h[:, self.config.image_token_len :]], dim=1
        )
        pos = (self.model.freqs_cos[: h.size(1)], self.model.freqs_sin[: h.size(1)])
        for layer in self.model.layers:
            h, _ = layer(h, pos, attention_mask=attention_mask.to(h.dtype))
        return self.model.norm(h)


class MiniMindVLA(nn.Module):
    """VLM 문맥에 cross-attention하는 작은 행동 Transformer."""

    def __init__(self, config: VLAConfig):
        super().__init__()
        self.config = config
        self.vlm = EducationalVLM(
            VLMConfig(**config.vlm_config), config.vision_model_path
        )
        d = config.expert_dim
        self.context_proj = nn.Linear(self.vlm.config.hidden_size, d)
        self.state_proj = nn.Linear(config.state_dim, d)
        self.action_in = nn.Linear(config.action_dim, d)
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.action_position = nn.Parameter(torch.randn(1, config.chunk_size, d) * 0.02)
        self.expert = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d,
                config.expert_heads,
                dim_feedforward=d * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            config.expert_layers,
        )
        self.action_out = nn.Linear(d, config.action_dim)
        if config.freeze_vlm:
            self.vlm.requires_grad_(False)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_vlm:
            self.vlm.eval()
        else:
            self.vlm.vision_encoder.eval()
        return self

    def encode_context(self, input_ids, attention_mask, pixel_values, state):
        """이미지·지시문·현재 상태를 memory [B,L+1,D]로 만든다."""
        c = self.config
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [B,L]")
        b, l = input_ids.shape
        if (
            state.shape != (b, c.state_dim)
            or pixel_values.ndim != 4
            or pixel_values.shape[:2] != (b, 3)
        ):
            raise ValueError("state [B,S] and RGB pixels [B,3,H,W] required")
        n = self.vlm.config.image_token_len
        marker = self.vlm.config.image_ids[0]
        # One contiguous prefix keeps image insertion unambiguous.
        if (
            l <= n
            or not (input_ids[:, :n] == marker).all()
            or (input_ids[:, n:] == marker).any()
        ):
            raise ValueError("Exactly one image token prefix is required")
        if not attention_mask[:, :n].all():
            raise ValueError("image prefix cannot be padded")
        if l > self.vlm.config.max_position_embeddings:
            raise ValueError("context too long")
        h = self.vlm(input_ids, attention_mask, pixel_values)
        return self.project_context(h, attention_mask, state)

    def project_context(self, hidden, attention_mask, state):
        """VLM 출력과 상태를 결합한다. 고정 VLM의 캐시도 이 경로를 사용한다."""
        memory = torch.cat(
            [self.context_proj(hidden), self.state_proj(state)[:, None]], dim=1
        )
        valid = torch.cat(
            [
                attention_mask.bool(),
                torch.ones(hidden.size(0), 1, dtype=torch.bool, device=hidden.device),
            ],
            dim=1,
        )
        return memory, valid

    def velocity(self, memory, valid, noisy_actions, time, action_mask=None):
        """dx/dt [B,H,A] 예측. 물리적 속도가 아니라 생성 시간에 대한 변화율이다."""
        d = self.config.expert_dim
        frequencies = torch.exp(
            torch.arange(d // 2, device=time.device, dtype=torch.float32)
            * (-math.log(10000) / (max(d // 2 - 1, 1)))
        )
        # 1000은 시간 embedding의 주파수 배율이며 생성 step 수가 아니다.
        phase = time[:, None] * frequencies[None] * 1000
        time_emb = self.time_mlp(
            torch.cat([phase.sin(), phase.cos()], dim=-1).to(memory.dtype)
        )
        x = self.action_in(noisy_actions) + time_emb[:, None] + self.action_position
        # All action positions see each other, unlike autoregressive language.
        x = self.expert(
            x,
            memory,
            memory_key_padding_mask=~valid,
            tgt_key_padding_mask=None if action_mask is None else ~action_mask.bool(),
        )
        return self.action_out(x)

    @staticmethod
    def flow_inputs(actions, noise, time):
        """x_t = (1-t)ε + ta, 그 경로의 미분 dx_t/dt = a-ε."""
        t = time[:, None, None]
        noisy_actions = (1 - t) * noise + t * actions
        target_velocity = actions - noise
        return noisy_actions, target_velocity

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        state,
        actions,
        action_mask=None,
        noise=None,
        time=None,
    ):
        """관측에서 문맥을 구한 뒤, 임의 시간의 속도 예측 MSE를 반환한다."""
        memory, valid = self.encode_context(
            input_ids, attention_mask, pixel_values, state
        )
        return self.flow_loss(memory, valid, actions, action_mask, noise, time)

    def flow_loss(
        self, memory, valid, actions, action_mask=None, noise=None, time=None
    ):
        """한 번의 속도 예측으로 학습한다. 학습 중에는 ODE 적분을 하지 않는다."""
        b = memory.size(0)
        c = self.config
        if actions.shape != (b, c.chunk_size, c.action_dim):
            raise ValueError("wrong action shape")
        if action_mask is None:
            action_mask = torch.ones(
                actions.shape[:2], device=actions.device, dtype=torch.bool
            )
        action_mask = action_mask.bool()
        if action_mask.shape != actions.shape[:2] or not action_mask.any(dim=1).all():
            raise ValueError("each sample needs valid actions")
        noise = torch.randn_like(actions) if noise is None else noise
        time = torch.rand(b, device=actions.device) if time is None else time
        if (
            noise.shape != actions.shape
            or time.shape != (b,)
            or not ((time >= 0) & (time <= 1)).all()
        ):
            raise ValueError("noise shape or time range invalid")
        # 패딩을 산술 연산 전에 지워 NaN/Inf가 backward로 새는 것도 방지한다.
        padding = ~action_mask[:, :, None]
        actions = actions.masked_fill(padding, 0)
        noise = noise.masked_fill(padding, 0)
        noisy, target = self.flow_inputs(actions, noise, time)
        pred = self.velocity(memory, valid, noisy, time, action_mask)
        # FP32에서 제곱하여 mixed precision의 FP16 overflow를 피한다.
        error = (pred.float() - target.float()).square().masked_fill(padding, 0)
        return error.sum() / (action_mask.sum() * c.action_dim)

    @torch.no_grad()
    def integrate_actions(self, memory, valid, noise, steps=10):
        """eval 모드의 문맥과 노이즈로 Euler 적분한다. 전체 청크를 매번 갱신한다."""
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        shape = (memory.size(0), self.config.chunk_size, self.config.action_dim)
        if noise.shape != shape:
            raise ValueError("wrong noise shape")

        actions = noise
        dt = 1.0 / steps
        for step in range(steps):
            time = torch.full((actions.size(0),), step / steps, device=actions.device)
            velocity = self.velocity(memory, valid, actions, time)
            actions = actions + dt * velocity
        return actions

    @torch.no_grad()
    def sample_actions(
        self, input_ids, attention_mask, pixel_values, state, noise=None, steps=10
    ):
        """① 문맥 1회 계산 → ② 노이즈 생성 → ③ 적분 → 행동 [B,H,A]."""
        was_training = self.training
        self.eval()
        try:
            memory, valid = self.encode_context(
                input_ids, attention_mask, pixel_values, state
            )
            shape = (input_ids.size(0), self.config.chunk_size, self.config.action_dim)
            if noise is None:
                noise = torch.randn(shape, device=state.device, dtype=state.dtype)
            return self.integrate_actions(memory, valid, noise, steps)
        finally:
            self.train(was_training)

    def load_vlm_weights(self, path):
        """Load the native VLM prefix; ignore only known discarded-layer weights."""
        weights = torch.load(path, map_location="cpu", weights_only=True)
        kept = len(self.vlm.model.layers)
        source_depth = self.config.vlm_config.get("num_hidden_layers", 8)
        discarded = {
            f"model.layers.{i}.{key}"
            for i in range(kept, source_depth)
            for key in self.vlm.model.layers[0].state_dict()
        }
        weights = {k: v for k, v in weights.items() if k not in discarded}
        _load_weights(self.vlm, weights, "vision_encoder.", "VLM")

    def save_checkpoint(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # SigLIP is frozen and supplied by its original local model directory.
        weights = {
            k: v
            for k, v in self.state_dict().items()
            if not k.startswith("vlm.vision_encoder.")
        }
        torch.save(
            {"format_version": 2, "config": asdict(self.config), "model": weights}, path
        )

    @classmethod
    def load_checkpoint(cls, path, vision_model_path=None):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format_version") != 2:
            raise ValueError(
                "Half-depth VLA requires format_version=2; retrain the action expert "
                "from native MiniMind-V weights"
            )
        config = VLAConfig(**data["config"])
        if vision_model_path is not None:
            config.vision_model_path = vision_model_path
        model = cls(config)
        _load_weights(model, data["model"], "vlm.vision_encoder.", "VLA")
        return model


def _load_weights(module, weights, external_prefix, name):
    """별도 폴더에서 읽는 SigLIP 외에는 모든 가중치가 정확히 있어야 한다."""
    expected = set(module.state_dict())
    missing = expected - set(weights)
    unexpected = set(weights) - expected
    if unexpected or any(not key.startswith(external_prefix) for key in missing):
        raise ValueError(
            f"Incompatible {name} checkpoint: missing={sorted(missing)}, extra={sorted(unexpected)}"
        )
    module.load_state_dict(weights, strict=not missing)
