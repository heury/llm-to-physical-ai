import math, torch, torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import MaskedLMOutput
from model.model_minimind import MiniMindConfig, MiniMindModel

def add_gumbel_noise(logits, temperature):
    if temperature == 0: return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((-torch.log(noise)) ** temperature)

class MiniMindDLLMConfig(MiniMindConfig):
    model_type = "minimind_dllm"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask_token_id = kwargs.get("mask_token_id", 27)
        self.mask_epsilon = kwargs.get("mask_epsilon", 0.001)

class MiniMindDLLMModel(MiniMindModel):
    def __init__(self, config: MiniMindDLLMConfig):
        super().__init__(config)
        for layer in self.layers: layer.self_attn.is_causal = False

    def add_noise_to_tokens(self, input_ids, t, eps=None, pad_token_id=0):
        batch_size, seq_len = input_ids.shape
        eps = eps if eps is not None else self.config.mask_epsilon
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask.unsqueeze(-1).expand(batch_size, seq_len)
        corruption_mask = torch.rand(batch_size, seq_len, device=input_ids.device) < p_mask
        corruption_mask = corruption_mask & (input_ids != pad_token_id)
        noisy_input_ids = torch.where(corruption_mask, self.config.mask_token_id, input_ids)
        return noisy_input_ids, corruption_mask, p_mask

class MiniMindForMaskedDiffusion(PreTrainedModel, GenerationMixin):
    config_class = MiniMindDLLMConfig
    def __init__(self, config: MiniMindDLLMConfig = None):
        self.config = config or MiniMindDLLMConfig()
        super().__init__(self.config)
        self.model = MiniMindDLLMModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, labels=None, corruption_mask=None, p_mask=None, n_valid=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        logits = self.lm_head(hidden_states).float()
        loss = None
        if labels is not None and corruption_mask is not None and p_mask is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1), reduction='none').view(labels.shape)
            denom = n_valid if n_valid is not None else corruption_mask.sum().clamp_min(1)
            loss = (loss[corruption_mask] / p_mask[corruption_mask]).sum() / denom
        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=hidden_states)

    def add_noise_to_tokens(self, input_ids, t, eps=None, pad_token_id=0):
        return self.model.add_noise_to_tokens(input_ids, t, eps, pad_token_id)

    @torch.inference_mode()
    def generate(self, inputs, max_new_tokens=128, temperature=0.5, top_k=50, steps=128, eos_token_id=None, tokenizer=None, cfg_scale=0.0, **kwargs):
        input_ids = kwargs.get("input_ids", inputs)
        bsz, prompt_len, device = input_ids.shape[0], input_ids.shape[1], input_ids.device
        mask_id = self.config.mask_token_id
        eos_id = eos_token_id or self.config.eos_token_id
        block_size = kwargs.get("block_size", max_new_tokens)

        T = prompt_len + max_new_tokens
        x = torch.full((bsz, T), eos_id, dtype=torch.long, device=device)
        x[:, :prompt_len] = input_ids
        x[:, prompt_len:] = mask_id
        unmasked_index = (x != mask_id)

        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = steps

        for b in range(num_blocks):
            block_end = min(prompt_len + (b + 1) * block_size, T)

            for i in range(steps_per_block):
                mask_index = (x == mask_id)
                mask_count = mask_index[:, :block_end].sum(-1).min().item()
                if mask_count == 0: break
                n_unmask = max(1, round(mask_count / (steps_per_block - i)))

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(input_ids=x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(input_ids=x).logits

                if top_k > 0: logits[logits < torch.topk(logits, top_k, dim=-1)[0][..., -1:]] = -float('inf')

                x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                x0_p[:, block_end:] = -float('inf')

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-float('inf'), device=device))

                for j in range(bsz):
                    _, idx = torch.topk(confidence[j], k=min(n_unmask, int(mask_count)))
                    x[j, idx] = x0[j, idx]

            if tokenizer and kwargs.get("stream", False):
                print(f"[Block {b+1}/{num_blocks}] {tokenizer.decode(x[0, prompt_len:], skip_special_tokens=False)}")

        return x
