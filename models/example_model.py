from typing import Optional

import torch
import torch.nn as nn
import transformers

from .example_config import ExampleConfig
from .example_layers import RMSNorm, SwiGLUMLP, FlashMultiHeadAttention, fused_linear_cross_entropy
from .default_layers import RotaryEmbedding
from .default_model import TransformerDecoderLayer, TransformerModel, TransformerForCausalLM


class ExampleTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, config: ExampleConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.self_attn = FlashMultiHeadAttention(config, layer_idx)
        self.mlp = SwiGLUMLP(config)
        self.pre_attention_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class ExampleTransformerModel(TransformerModel):
    def __init__(self, config: ExampleConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList([ExampleTransformerDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_embed = RotaryEmbedding(config)
        self.gradient_checkpointing = False

        self.post_init()


class ExampleTransformerForCausalLM(TransformerForCausalLM):
    def __init__(self, config: ExampleConfig):
        super().__init__(config)
        self.config = config
        self.model = ExampleTransformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[transformers.cache_utils.Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
        )
        hidden_states = outputs.last_hidden_state
        logits = None
        loss = None

        if labels is not None:
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
            shift_labels = shift_labels.view(-1)

            loss = fused_linear_cross_entropy(shift_hidden_states, shift_labels, self.lm_head.weight)
        else:
            logits = self.lm_head(hidden_states)

        return transformers.modeling_outputs.CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=None,
        )
