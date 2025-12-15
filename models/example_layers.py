from typing import Optional

import torch
import torch.nn.functional as F
import transformers

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import pad_input, unpad_input

from liger_kernel.transformers.functional import liger_cross_entropy
from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

from .example_config import ExampleConfig
from .default_layers import MultiHeadAttention, apply_rope

RMSNorm = LigerRMSNorm
SwiGLUMLP = LigerSwiGLUMLP
GeGLUMLP = LigerGEGLUMLP
cross_entropy = liger_cross_entropy


def fused_linear_cross_entropy(
    shifted_hidden_states: torch.Tensor,
    shifted_labels: torch.Tensor,
    lm_head_weight: torch.Tensor,
):
    lce = LigerFusedLinearCrossEntropyLoss()
    loss = lce(lm_head_weight, shifted_hidden_states, shifted_labels)
    return loss


class FlashMultiHeadAttention(MultiHeadAttention):
    """
    Flash Attention 2 replacement for the default Multi-Head Attention layer.
    All Flash-specific helpers are kept within this class to make the example
    self-contained.
    """

    @staticmethod
    def _get_unpad_data(attention_mask: torch.Tensor):
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        return indices, cu_seqlens, max_seqlen_in_batch

    @classmethod
    def _unpad_input(
        cls,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = cls._get_unpad_data(attention_mask)

        if key_layer.shape[1] > (seq_len := attention_mask.shape[-1]):
            key_layer = key_layer[:, :seq_len, :, :]
            value_layer = value_layer[:, :seq_len, :, :]

        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        def _index_first_axis(tensor, indices):
            reshaped = tensor.reshape(-1, *tensor.shape[2:])
            return reshaped[indices]

        key_layer = _index_first_axis(key_layer, indices_k)
        value_layer = _index_first_axis(value_layer, indices_k)

        if query_length == kv_seq_len:
            query_layer = _index_first_axis(query_layer, indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q, *_ = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    @classmethod
    def _flash_attention(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = True,
        training: bool = False,
    ):
        query_length = query.shape[1]

        if attention_mask is not None:
            q, k, v, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = (
                cls._unpad_input(query, key, value, attention_mask, query_length)
            )

            attn_output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=dropout if training else 0.0,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output, indices_q, query.shape[0], query_length)
        else:
            attn_output = flash_attn_func(
                query,
                key,
                value,
                dropout_p=dropout if training else 0.0,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output


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

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        cos, sin = position_embeddings
        query_states_rope = query_states.transpose(1, 2)
        key_states_rope = key_states.transpose(1, 2)
        apply_rope(query_states_rope, key_states_rope, cos, sin)
        query_states = query_states_rope.transpose(1, 2)
        key_states = key_states_rope.transpose(1, 2)

        if past_key_values is not None:
            key_cache = key_states.transpose(1, 2)
            value_cache = value_states.transpose(1, 2)

            key_cache, value_cache = past_key_values.update(
                key_cache,
                value_cache,
                self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )

            key_states = key_cache.transpose(1, 2)
            value_states = value_cache.transpose(1, 2)

        attn_output = self._flash_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            dropout=self.dropout,
            softmax_scale=self.scaling,
            causal=True,
            training=self.training,
        )

        return self.o_proj(attn_output.reshape(*input_shape, -1).contiguous())


__all__ = [
    "RMSNorm",
    "SwiGLUMLP",
    "GeGLUMLP",
    "MultiHeadAttention",
    "FlashMultiHeadAttention",
    "cross_entropy",
    "fused_linear_cross_entropy",
]
