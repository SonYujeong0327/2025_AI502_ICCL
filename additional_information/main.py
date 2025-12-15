import logging
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoTokenizer, HfArgumentParser, Trainer, TrainingArguments, set_seed

from dataset import prepare_pretrain_dataset
from models.default_model import TransformerConfig, TransformerForCausalLM


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    base_model_name_or_path: str = field(
        default="meta-llama/Llama-3.2-1B",
    )


@dataclass
class DataArguments:
    dataset_name_or_path: str = field(
        default="HuggingFaceFW/fineweb",
        metadata={"help": "Dataset name or local path containing raw text data."},
    )
    dataset_subset: Optional[str] = field(
        default="sample-10BT",
        metadata={"help": "Optional dataset configuration or subset name."},
    )
    text_column_name: str = field(default="text", metadata={"help": "Column with raw text to tokenize."})
    max_length: int = field(default=1024, metadata={"help": "Block size after packing tokenized text."})


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(level=logging.INFO)
    if training_args.logging_dir:
        logger.setLevel(logging.INFO)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = prepare_pretrain_dataset(
        tokenizer=tokenizer,
        dataset_name_or_path=data_args.dataset_name_or_path,
        dataset_subset=data_args.dataset_subset,
        max_length=data_args.max_length,
        text_column_name=data_args.text_column_name,
        train_sample_size=data_args.train_sample_size if data_args.train_sample_size > 0 else -1,
        eval_sample_size=data_args.eval_sample_size if data_args.eval_sample_size > 0 else -1,
        cache_path=data_args.cache_path,
        num_proc=data_args.num_proc,
        eval_split=data_args.eval_split,
        eval_ratio=data_args.eval_ratio,
        seed=data_args.seed,
        streaming=data_args.streaming,
    )

    logger.info("Loaded training dataset with %d samples.", len(train_dataset))
    logger.info("Loaded evaluation dataset with %d samples.", len(eval_dataset))

    num_key_value_heads = model_args.num_key_value_heads or model_args.num_attention_heads
    config = TransformerConfig(
        base_model_name_or_path=None,
        vocab_size=len(tokenizer),
        hidden_size=model_args.hidden_size,
        intermediate_size=model_args.intermediate_size,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=model_args.num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=model_args.max_position_embeddings,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
    )

    model = TransformerForCausalLM(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn_for_pretrain,
        tokenizer=tokenizer,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics if hasattr(train_result, "metrics") else {}
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
