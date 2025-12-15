from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import transformers

from .default_config import TransformerConfig
from .default_layers import (
    RMSNorm as DefaultRMSNorm,
    SwiGLUMLP as DefaultSwiGLUMLP,
    MultiHeadAttention as DefaultMultiHeadAttention,
    RotaryEmbedding,
    fused_linear_cross_entropy,
)
from .default_model import TransformerPretrainedModel
from .example_layers import (
    RMSNorm as ExampleRMSNorm,
    SwiGLUMLP as ExampleSwiGLUMLP,
    FlashMultiHeadAttention,
)


@dataclass
class AdapterSchedule:
    """
    Linear schedule that anneals the contribution of the base (default) module
    towards the accelerated example module.

    base_weight progresses from `base_start` to `base_end` across `total_steps`.
    example_weight is computed as (1 - base_weight).
    """

    total_steps: int
    base_start: float = 0.9
    base_end: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.base_start <= 1.0 or not 0.0 <= self.base_end <= 1.0:
            raise ValueError("Base weights must lie in [0, 1].")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        self._current_step: int = 0

    def state_dict(self) -> dict:
        return {"current_step": self._current_step}

    def load_state_dict(self, state: dict) -> None:
        self._current_step = int(state.get("current_step", 0))

    def set_step(self, step: int) -> None:
        self._current_step = max(0, step)

    def increment(self, steps: int = 1) -> None:
        self._current_step = max(0, self._current_step + steps)

    def weights(self) -> Tuple[float, float]:
        progress = min(1.0, self._current_step / self.total_steps)
        base_weight = self.base_start + (self.base_end - self.base_start) * progress
        base_weight = min(1.0, max(0.0, base_weight))
        example_weight = 1.0 - base_weight
        return base_weight, example_weight


class HybridTransformerDecoderLayer(nn.Module):
    """
    Decoder block that linearly interpolates between the default implementation
    (baseline PyTorch eager kernels) and the optimized example implementation
    (Flash Attention + fused MLP) according to an AdapterSchedule.
    """

    def __init__(self, config: TransformerConfig, layer_idx: int, schedule: AdapterSchedule):
        super().__init__()
        self.layer_idx = layer_idx
        self.schedule = schedule

        # Baseline path (default model components)
        self.base_self_attn = DefaultMultiHeadAttention(config, layer_idx)
        self.base_mlp = DefaultSwiGLUMLP(config)
        self.base_pre_norm = DefaultRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.base_post_norm = DefaultRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Optimized path (example components)
        self.example_self_attn = FlashMultiHeadAttention(config, layer_idx)
        self.example_mlp = ExampleSwiGLUMLP(config)
        self.example_pre_norm = ExampleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.example_post_norm = ExampleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[transformers.cache_utils.Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if past_key_values is not None:
            raise NotImplementedError("Hybrid layers do not currently support KV-cache usage.")

        base_weight, example_weight = self.schedule.weights()

        # Attention branch
        base_attn_input = self.base_pre_norm(hidden_states)
        example_attn_input = self.example_pre_norm(hidden_states)

        base_attn_output = self.base_self_attn(
            hidden_states=base_attn_input,
            attention_mask=attention_mask,
            past_key_values=None,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        example_attn_output = self.example_self_attn(
            hidden_states=example_attn_input,
            attention_mask=attention_mask,
            past_key_values=None,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        attn_output = base_weight * base_attn_output + example_weight * example_attn_output
        hidden_states = hidden_states + attn_output

        # Feed-forward branch
        base_mlp_input = self.base_post_norm(hidden_states)
        example_mlp_input = self.example_post_norm(hidden_states)

        base_mlp_output = self.base_mlp(base_mlp_input)
        example_mlp_output = self.example_mlp(example_mlp_input)

        mlp_output = base_weight * base_mlp_output + example_weight * example_mlp_output
        hidden_states = hidden_states + mlp_output

        return hidden_states


class HybridTransformerModel(TransformerPretrainedModel):
    """
    Transformer decoder stack that mixes baseline and optimized submodules.
    """

    def __init__(self, config: TransformerConfig, schedule: AdapterSchedule):
        super().__init__(config)
        self.adapter_schedule = schedule

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList(
            HybridTransformerDecoderLayer(config, i, schedule) for i in range(config.num_hidden_layers)
        )
        self.norm = DefaultRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                dtype=torch.long,
                device=inputs_embeds.device,
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

    def set_annealing_step(self, step: int) -> None:
        self.adapter_schedule.set_step(step)

    def increment_annealing_step(self, steps: int = 1) -> None:
        self.adapter_schedule.increment(steps)

    def get_current_weights(self) -> Tuple[float, float]:
        return self.adapter_schedule.weights()


class ExampleAdapterForCausalLM(TransformerPretrainedModel, transformers.generation.GenerationMixin):
    """
    Causal LM that anneals from the baseline default transformer to the optimized
    example transformer implementation over the course of pretraining.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: TransformerConfig, schedule: Optional[AdapterSchedule] = None):
        super().__init__(config)
        self.adapter_schedule = schedule or AdapterSchedule(total_steps=1000, base_start=0.9, base_end=0.0)
        self.model = HybridTransformerModel(config, self.adapter_schedule)
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

        loss = None
        logits = None
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

    def set_annealing_step(self, step: int) -> None:
        self.model.set_annealing_step(step)

    def increment_annealing_step(self, steps: int = 1) -> None:
        self.model.increment_annealing_step(steps)

    def get_current_weights(self) -> Tuple[float, float]:
        return self.model.get_current_weights()

    def state_dict(self):
        state = super().state_dict()
        state["_adapter_schedule"] = self.adapter_schedule.state_dict()
        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        schedule_state = state_dict.pop("_adapter_schedule", None)
        super().load_state_dict(state_dict, strict=strict)
        if schedule_state is not None:
            self.adapter_schedule.load_state_dict(schedule_state)
