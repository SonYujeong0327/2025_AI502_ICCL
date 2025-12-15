import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from typing import Optional, Union

from .default_config import TransformerConfig
from .default_layers import RMSNorm, SwiGLUMLP, fused_linear_cross_entropy, MultiHeadAttention, RotaryEmbedding


class TransformerDecoderLayer(transformers.modeling_layers.GradientCheckpointingLayer):
    def __init__(self, config: TransformerConfig, layer_idx):
        super().__init__()

        self.self_attn = MultiHeadAttention(config, layer_idx)
        self.mlp = SwiGLUMLP(config)
        self.pre_attention_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[transformers.cache_utils.Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
    ):
        hidden_states = hidden_states + self.self_attn(
                                            hidden_states=self.pre_attention_layer_norm(hidden_states),
                                            attention_mask=attention_mask,
                                            past_key_values=past_key_values,
                                            cache_position=cache_position,
                                            position_embeddings=position_embeddings,
                                        )
        hidden_states = hidden_states + self.mlp(self.post_attention_layer_norm(hidden_states))
        return hidden_states

class TransformerPretrainedModel(transformers.PreTrainedModel):
    config_class = TransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

class TransformerModel(TransformerPretrainedModel):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList([TransformerDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_embed = RotaryEmbedding(config)
        self.gradient_checkpointing = False

        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[transformers.cache_utils.Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify either input_ids or input_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if use_cache and past_key_values is None:
            past_key_values = transformers.cache_utils.DynamicCache(config=self.config)
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
                dtype=torch.long, device=inputs_embeds.device
            )
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_embed(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)

        return transformers.modeling_outputs.BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

class TransformerForCausalLM(TransformerPretrainedModel, transformers.generation.GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.model = TransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[transformers.cache_utils.Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
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
