from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
import transformers
import logging
import math

# from flash_attn import flash_attn_func, flash_attn_varlen_func
# from flash_attn.bert_padding import pad_input, unpad_input
from transformers.modeling_flash_attention_utils import _flash_attention_forward, flash_attn_supports_top_left_mask

from liger_kernel.transformers.functional import liger_cross_entropy
from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

from models.default_layers import apply_rope, RotaryEmbedding
from .config import DTConfig

RMSNorm = LigerRMSNorm
SwiGLUMLP = LigerSwiGLUMLP
GeGLUMLP = LigerGEGLUMLP
cross_entropy = liger_cross_entropy
logger = logging.getLogger(__name__)

def lambda_init_fn(layer_idx):
    return 0.8 - 0.6 * math.exp(-0.3 * layer_idx)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This function repeats the Key and Value tensors for Grouped-Query Attention (GQA).
    It ensures that the number of query heads matches the number of Key/Value head groups.
    """
    # TODO: Implement the logic to repeat the Key and Value tensors n_rep times.
    # Hint: You need to efficiently transform the shape of hidden_states from
    # (batch, kv_head_num, slen, head_dim) to (batch, kv_head_num * n_rep, slen, head_dim).
    # You can use torch.repeat_interleave or a combination of reshape/expand/repeat.
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def _rope_(
    config: Optional[DTConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
) -> tuple["torch.Tensor", float]:
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor

def fused_linear_cross_entropy(
    shifted_hidden_states: torch.Tensor,
    shifted_labels: torch.Tensor,
    lm_head_weight: torch.Tensor
):
    lce = LigerFusedLinearCrossEntropyLoss()
    loss = lce(lm_head_weight, shifted_hidden_states, shifted_labels)
    return loss

class DTMLP(SwiGLUMLP):
    pass

class DTAttention(nn.Module):
    def __init__(self, config: DTConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.head_num = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.head_num)
        self.kv_head_num = config.num_key_value_heads
        self.num_key_value_groups = self.head_num // self.kv_head_num

        self.is_causal = True

        # TODO: Initialize the nn.Linear layers for the Query, Key, Value, and Output projections.
        # You should apply bias according to the config.attention_bias setting.
        self.q_proj = nn.Linear(self.hidden_size, self.head_num * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_head_num * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_head_num * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.head_num * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.lambda_init = lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(torch.normal(0, config.lambda_std_dev, size=(self.head_dim,)))
        self.lambda_k1 = nn.Parameter(torch.normal(0, config.lambda_std_dev, size=(self.head_dim,)))
        self.lambda_q2 = nn.Parameter(torch.normal(0, config.lambda_std_dev, size=(self.head_dim,)))
        self.lambda_k2 = nn.Parameter(torch.normal(0, config.lambda_std_dev, size=(self.head_dim,)))
        # TODO : implment groupnorm of differential transformers
        self.groupnorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[transformers.cache_utils.Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, target_len, _ = hidden_states.size()
        q_len = target_len

        # TODO 1: Project the input `hidden_states` into Query, Key, and Value using `q_proj`, `k_proj`, and `v_proj`.
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # TODO 2: Reshape the Query, Key, and Value tensors for Multi-head Attention.
        # First, view them as (bsz, q_len, num_heads, head_dim), then transpose the axes
        # to (bsz, num_heads, q_len, head_dim) for the attention calculation.
        query_states = query_states.view(bsz, q_len, self.head_num, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.kv_head_num, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.kv_head_num, self.head_dim).transpose(1, 2)

        # TODO 3: Apply Rotary Position Embedding (RoPE) to the Query and Key states using the `apply_rope` function.
        cos, sin = position_embeddings
        # ... write your code here ...
        
        if cos.ndim == 4:
            if cos.shape[1] > q_len: # (B, L, H, D) 형태일 때
                cos = cos[:, -q_len:, :, :]
                sin = sin[:, -q_len:, :, :]
            if cos.shape[2] > q_len: # (B, H, L, D) 형태일 때
                cos = cos[:, :, -q_len:, :]
                sin = sin[:, :, -q_len:, :]
        elif cos.ndim == 3: # (B, L, D) 형태일 때
            if cos.shape[1] > q_len:
                cos = cos[:, -q_len:, :]
                sin = sin[:, -q_len:, :]
        
        query_states, key_states = apply_rope(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )

        # TODO 4: Repeat the Key and Value heads for GQA using the `repeat_kv` function.
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # TODO 5: Construct the special Value tensor for the Differential Transformer.
        # Hint: The original value_states has shape (bsz, num_heads, q_len, head_dim). 
        # Split it into two along dim=1 (the head dimension), then concatenate them along dim=-1 (the feature dimension). 
        # Finally, repeat the result to double the number of heads.
        v1, v2 = value_states.chunk(2, dim=1)
        value_states = torch.cat([v1, v2], dim=-1)
        value_states = repeat_kv(value_states, 2)

        # TODO 6: Calculate the scores for Scaled Dot-Product Attention (matrix multiplication of Query and Key).
        # Don't forget to scale by the square root of `self.head_dim`.
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # TODO 7: If an `attention_mask` is provided, apply it to the attention scores (Causal Masking).
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            
            if attention_mask.size(-1) > key_states.size(2):
                attention_mask = attention_mask[..., :key_states.size(2)]
                
            attn_weights = attn_weights + attention_mask

        # TODO 8: Apply the Softmax function to the attention scores to get probabilities, then apply dropout.
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # TODO 9: Calculate `lambda_full`, the core parameter of the Differential Transformer.
        # Hint: Use parameters like `lambda_q1`, `lambda_k1`, etc.
        lambda_1 = torch.einsum('d,d->', self.lambda_q1, self.lambda_k1)
        lambda_2 = torch.einsum('d,d->', self.lambda_q2, self.lambda_k2)
        lambda_full = torch.exp(lambda_1 + lambda_2) + self.lambda_init

        # TODO 10: Compute the final attention output by multiplying the attention weights with the Value tensor.
        attn_output = torch.matmul(attn_weights, value_states)

        # TODO 11: Apply the Differential Transformer's formula to transform the `attn_output`.
        # Hint: Split `attn_output` into two chunks, combine them using `lambda_full`, and then apply `groupnorm`.
        # The result of TODO 5 gives us (B, H, L, 2D). We split back to D to compute the difference.
        o1, o2 = attn_output.chunk(2, dim=-1)
        attn_output = o1 - lambda_full * o2
        attn_output = self.groupnorm(attn_output)
        # Apply the scaling factor (1 - lambda_init) as per Diff Transformer paper
        attn_output = attn_output * (1 - self.lambda_init)

        # TODO 12: Reshape the final `attn_output` back to (bsz, q_len, hidden_size) and pass it through the `o_proj` layer.
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

class DTFlashAttention2(DTAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = flash_attn_supports_top_left_mask()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[transformers.cache_utils.Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, None]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # 1. Projection
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 2. Reshape to (Batch, Seq, Head, Dim)
        query_states = query_states.view(bsz, q_len, self.head_num, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.kv_head_num, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.kv_head_num, self.head_dim)

        # 3. RoPE
        query_states = query_states.transpose(1, 2) # (B, H, L, D)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        cos, sin = position_embeddings
        
        if cos.shape[-2] > q_len:
            cos = cos[..., -q_len:, :]
            sin = sin[..., -q_len:, :]
        elif cos.ndim == 2 and cos.shape[0] > q_len: # (Seq, Dim) 형태일 때
            cos = cos[-q_len:, :]
            sin = sin[-q_len:, :]
        
        query_states, key_states = apply_rope(query_states, key_states, cos, sin)
        
        # KV Cache Update
        if past_key_values is not None:
             key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )

        # # 4. Flash Attention 입력을 위해 다시 (Batch, Seq, Head, Dim)으로 변환
        # query_states = query_states.transpose(1, 2)
        # key_states = key_states.transpose(1, 2)
        # value_states = value_states.transpose(1, 2)
        
        # # Value state가 Cache에서 나오면서 Transpose 되어 있을 수 있으므로 확인
        # # if value_states.size(1) != q_len: 
        # #      value_states = value_states.transpose(1, 2)

        # query_states = query_states.contiguous()
        # key_states = key_states.contiguous()
        # value_states = value_states.contiguous()

        # # 5. Differential Transformer 분할 (Chunk)
        # q1, q2 = query_states.chunk(2, dim=2)
        # k1, k2 = key_states.chunk(2, dim=2)
        # v1, v2 = value_states.chunk(2, dim=2)

        # q1, q2 = q1.contiguous(), q2.contiguous()
        # k1, k2 = k1.contiguous(), k2.contiguous()
        # v1, v2 = v1.contiguous(), v2.contiguous()

        # # 6. Flash Attention Func 직접 호출
        # dropout_p = self.attention_dropout if self.training else 0.0
        
        # attn_output1 = F.scaled_dot_product_attention(
        #     q1, k1, v1, dropout_p=dropout_p, is_causal=self.is_causal)
        # attn_output2 = F.scaled_dot_product_attention(
        #     q2, k2, v2, dropout_p=dropout_p, is_causal=self.is_causal)

        # # 7. Lambda 결합 (Differential Logic)
        # lambda_1 = torch.einsum("d,d->", self.lambda_q1, self.lambda_k1)
        # lambda_2 = torch.einsum("d,d->", self.lambda_q2, self.lambda_k2)
        # lambda_full = torch.exp(lambda_1 + lambda_2) + self.lambda_init

        # # attn_output: (B, L, H/2, D)
        # diff_output = attn_output1 - lambda_full * attn_output2
        
        # # GroupNorm & Scaling
        # diff_output = self.groupnorm(diff_output)
        # diff_output = diff_output * (1.0 - self.lambda_init)

        # # 8. Output Projection 복구
        # diff_output = diff_output.transpose(1, 2)
        # diff_output = repeat_kv(diff_output, 2) 
        # diff_output = diff_output.transpose(1, 2).contiguous()

        # diff_output = diff_output.reshape(bsz, q_len, self.hidden_size)
        # attn_output = self.o_proj(diff_output)

        # return attn_output
    
        # 4. Flash Attention 입력을 위해 (Batch, Head, Seq, Dim) 순서 유지
        # SDPA는 기본적으로 (B, H, L, D)를 기대합니다.
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # 5. Differential Transformer 분할 (Head 차원인 dim=1을 분할)
        # query_states: (B, H, L, D) -> q1, q2: (B, H/2, L, D)
        q1, q2 = query_states.chunk(2, dim=1)
        k1, k2 = key_states.chunk(2, dim=1)
        v1, v2 = value_states.chunk(2, dim=1)

        # 6. SDPA (Standard PyTorch Attention)
        # 결과 attn_output은 (B, H/2, L, D) 형태가 됩니다.
        dropout_p = self.attention_dropout if self.training else 0.0
        
        attn_output1 = F.scaled_dot_product_attention(
            q1, k1, v1, dropout_p=dropout_p, is_causal=self.is_causal)
        attn_output2 = F.scaled_dot_product_attention(
            q2, k2, v2, dropout_p=dropout_p, is_causal=self.is_causal)

        # 7. Differential Transformer 계산
        # diff_output: (B, H/2, L, D)
        lambda_1 = torch.einsum("d,d->", self.lambda_q1, self.lambda_k1)
        lambda_2 = torch.einsum("d,d->", self.lambda_q2, self.lambda_k2)
        lambda_full = torch.exp(lambda_1 + lambda_2) + self.lambda_init

        diff_output = attn_output1 - lambda_full * attn_output2
        
        # 8. Output Projection 복구 (가장 중요한 부분)
        # 현재 diff_output: (Batch, Head/2, Seq, Dim)
        
        # 8-1. GroupNorm을 위해 (Batch, Seq, Head/2, Dim)으로 일시 변환
        diff_output = diff_output.transpose(1, 2).contiguous() # (B, L, H/2, D)
        diff_output = self.groupnorm(diff_output)
        diff_output = diff_output * (1.0 - self.lambda_init)

        # 8-2. 다시 Head를 복구 (Head/2 -> Head)
        # repeat_kv가 dim=2(Head 차원)를 복제하도록 순서를 맞춰줍니다.
        # 현재 상태: (B, L, H/2, D)
        diff_output = repeat_kv(diff_output, 2) # (B, L, Head, D)
        
        # 8-3. 최종 결합 (Batch, Seq, Hidden_Size)
        # reshape 하기 전의 size를 확인하면 (16, 1, 32, 16) 이어야 합니다.
        # (16 * 1 * 32 * 16 = 8192 원소)
        diff_output = diff_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(diff_output)

        return attn_output

class DTRotaryEmbedding(RotaryEmbedding):
    def __init__(self, config: DTConfig, device=None):
        nn.Module.__init__(self)
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = _rope_
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq


__all__ = [
    "RMSNorm",
    "SwiGLUMLP",
    "GeGLUMLP",
    "DTAttention",
    "DTRotaryEmbedding",
    "apply_rope",
    "cross_entropy",
    "fused_linear_cross_entropy",
]