import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DTConfig
from .layers import RMSNorm, SwiGLUMLP, DTRotaryEmbedding, DTAttention, DTFlashAttention2
from models.default_model import TransformerDecoderLayer, TransformerModel, TransformerForCausalLM

DT_ATTENTION_CLASSES = {
    "eager": DTAttention,
    "flash_attention_2": DTFlashAttention2,
}

class DTDecoderLayer(TransformerDecoderLayer):
    def __init__(self, config: DTConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.self_attn = DT_ATTENTION_CLASSES[config.attention_implementation](config=config, layer_idx=layer_idx)
        self.mlp = SwiGLUMLP(config)
        self.pre_attention_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

class DTModel(TransformerModel):
    def __init__(self, config: DTConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList([DTDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_embed = DTRotaryEmbedding(config)
        self.gradient_checkpointing = False
        
        self.post_init()

class DTForCausalLM(TransformerForCausalLM):
    def __init__(self, config: DTConfig):
        super().__init__(config)
        self.config = config
        self.model = DTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()