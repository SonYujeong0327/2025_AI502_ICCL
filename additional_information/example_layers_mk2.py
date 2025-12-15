import inspect
import logging

from functools import partial
from types import MethodType
from typing import Callable
from typing import Optional

import transformers

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import pad_input, unpad_input

from .example_config import ExampleConfig
from .example_layers import RMSNorm, SwiGLUMLP
from .default_layers import MultiHeadAttention, apply_rope

def _unpad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
):

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices_k = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch_k = seqlens_in_batch.max().item()
    cu_seqlens_k = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

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
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )
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

class FlashMultiHeadAttention(MultiHeadAttention):
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
        
        # Handle KV cache
        if past_key_values is not None:
            # Cache expects (batch, num_heads, seq_len, head_dim)
            key_cache = key_states.transpose(1, 2)
            value_cache = value_states.transpose(1, 2)
            
            key_cache, value_cache = past_key_values.update(
                key_cache, value_cache, self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position}
            )
            
            # Convert back to Flash Attention format
            key_states = key_cache.transpose(1, 2)
            value_states = value_cache.transpose(1, 2)
        
        
        query_length = query_states.shape[1]
        

        if attention_mask is not None:
            q, k, v, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = _unpad_input(query_states, key_states, value_states, attention_mask, query_length)
            
            # Use varlen version for padded sequences
            attn_output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.scaling,
                causal=self.is_causal,
            )
            
            # Pad output back to original shape
            attn_output = pad_input(attn_output, indices_q, query_states.shape[0], query_length)
        else:
            # No padding: use standard flash_attn_func
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.scaling,
                causal=self.is_causal,
            )

        return self.o_proj(attn_output.reshape(*input_shape, -1).contiguous())
