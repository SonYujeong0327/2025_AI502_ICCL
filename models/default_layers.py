from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from .default_config import TransformerConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + self.eps)
        return hidden_states * inv_rms * self.weight


class SwiGLUMLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(F.silu(gate) * up)


class GeGLUMLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(F.gelu(gate, approximate="tanh") * up)


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels, reduction="mean")


def fused_linear_cross_entropy(
    shifted_hidden_states: torch.Tensor,
    shifted_labels: torch.Tensor,
    lm_head_weight: torch.Tensor,
) -> torch.Tensor:
    logits = F.linear(shifted_hidden_states.float(), lm_head_weight.float())
    loss = F.cross_entropy(logits, shifted_labels, reduction="mean")
    return loss.to(dtype=shifted_hidden_states.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim=1
) -> None:
    
    cos = cos.unsqueeze(unsqueeze_dim) 
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (query_states * cos) + (_rotate_half(query_states) * sin)
    k_embed = (key_states * cos) + (_rotate_half(key_states) * sin)
    
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig, layer_idx: int, is_causal: bool = True):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.head_num = config.num_attention_heads
        self.kv_head_num = config.num_key_value_heads

        self.scaling = self.head_dim ** -0.5
        self.is_causal = is_causal
        self.dropout = config.attention_dropout
        self.num_kv_groups = self.head_num // self.kv_head_num

        self.q_proj = nn.Linear(config.hidden_size, self.head_num * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_head_num * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_head_num * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.head_num * self.head_dim, config.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[transformers.cache_utils.Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rope(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )

        if self.num_kv_groups > 1:
            batch, num_key_value_heads, slen, head_dim = key_states.shape
            key_states = key_states[:, :, None, :, :].expand(batch, num_key_value_heads, self.num_kv_groups, slen, head_dim)
            key_states = key_states.reshape(batch, num_key_value_heads * self.num_kv_groups, slen, head_dim)

            value_states = value_states[:, :, None, :, :].expand(batch, num_key_value_heads, self.num_kv_groups, slen, head_dim)
            value_states = value_states.reshape(batch, num_key_value_heads * self.num_kv_groups, slen, head_dim)

        attn_weight = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, : key_states.shape[-2]].bool()
            attn_weight = attn_weight.masked_fill(~attention_mask, float("-inf"))

        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.attn_dropout(attn_weight)

        attn_output = torch.matmul(attn_weight, value_states)

        return self.o_proj(attn_output.reshape(*input_shape, -1).contiguous())


class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: TransformerConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @transformers.modeling_rope_utils.dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


__all__ = [
    "RMSNorm",
    "SwiGLUMLP",
    "GeGLUMLP",
    "MultiHeadAttention",
    "RotaryEmbedding",
    "apply_rope",
    "cross_entropy",
    "fused_linear_cross_entropy",
]
