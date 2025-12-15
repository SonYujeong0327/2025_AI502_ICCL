import transformers
from .default_config import TransformerConfig

logger = transformers.logging.get_logger(__name__)


class ExampleConfig(TransformerConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
