import transformers

logger = transformers.logging.get_logger(__name__)

class TransformerConfig(transformers.configuration_utils.PretrainedConfig):
    # Copy from transformers.models.llama.configuration_llama.LlamaConfig
    # DO NOT MODIFY THIS CLASS FOR SANITY CHECK AND COMMON LAYERS
    model_type = "DefaultTransformer"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        base_model_name_or_path=None,
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_dropout=0.0,
        hidden_act="silu",

        attention_implementation="eager",
        mlp_implementation="swiglu",
        **kwargs,
    ):
        self.base_model_name_or_path = base_model_name_or_path
        if base_model_name_or_path is not None:
            init_kwargs = {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": num_attention_heads,
                "num_key_value_heads": num_key_value_heads,
                "max_position_embeddings": max_position_embeddings,
                "initializer_range": initializer_range,
                "rms_norm_eps": rms_norm_eps,
                "use_cache": use_cache,
                "pad_token_id": pad_token_id,
                "bos_token_id": bos_token_id,
                "eos_token_id": eos_token_id,
                "tie_word_embeddings": tie_word_embeddings,
                "rope_theta": rope_theta,
                "rope_scaling": rope_scaling,
                "attention_dropout": attention_dropout,

                # "attention_implementation": attention_implementation,
                # "mlp_implementation": mlp_implementation,
                **kwargs,
            }
            base_config = transformers.AutoConfig.from_pretrained(base_model_name_or_path)
            logger.info(f"Loading base model config from {base_model_name_or_path}: {base_config} and override with given args")
            for key, value in base_config.to_dict().items():
                if key in init_kwargs and init_kwargs[key] != value:
                    logger.info(f"Overriding {key}: {init_kwargs[key]} -> {value}")
                    init_kwargs[key] = value
            self.__init__(**init_kwargs)
            return
        
        assert hidden_size % num_attention_heads == 0, "`hidden_size` must be divisible by `num_attention_heads`"
        assert num_attention_heads % num_key_value_heads == 0, "`num_attention_heads` must be divisible by `num_key_value_heads`"

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.hidden_act=hidden_act


        self.attention_implementation = attention_implementation
        # self.mlp_implementation = mlp_implementation

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        transformers.modeling_rope_utils.rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )