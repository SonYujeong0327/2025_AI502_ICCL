import transformers
from models.default_config import TransformerConfig

logger = transformers.logging.get_logger(__name__)

class DTConfig(TransformerConfig):
    model_type = "dt_model"
    def __init__(
        self,
        vocab_size=102400,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        attention_bias=False,
        lambda_std_dev=0.1,
        attention_implementation="eager",
        # attention_implementation="flash_attention_2",
        
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            **kwargs,
        )
        self.attention_bias = attention_bias
        self.lambda_std_dev = lambda_std_dev
        self.attention_implementation = attention_implementation