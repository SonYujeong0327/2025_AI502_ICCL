import logging
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)

import json
import torch
from train.dataset import prepare_pretrain_dataset
from models.dtmodel import DTForCausalLM, DTConfig

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    base_model_name_or_path: str = field(
        default="/mnt/raid5/kangjh/TA2025_2/release/assignment_2025/output/base_model",
        metadata={"help": "Base model checkpoint used for both data processing and model vocab sizing."},
    )
    model_config_path: str = field(
        default="config/debug.json",
        metadata={"help": "Optional path to model config file (if not using pretrained model)."},
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
    max_length: int = field(default=4096, metadata={"help": "Block size after packing tokenized text."})

def rounded_string(target: int) -> int:
    if target < 2**10:
        return target
    elif target < 2**20:
        return f"{target / 2**10:3.2f}K"
    elif target < 2**30:
        return f"{target / 2**20:3.2f}M"
    elif target < 2**40:
        return f"{target / 2**30:3.2f}G"
    elif target < 2**50:
        return f"{target / 2**40:3.2f}T"
    elif target < 2**60:
        return f"{target / 2**50:3.2f}P"
    else:
        return f"{target / 2**60:3.2f}E"

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-4")
    
    config = DTConfig(
        vocab_size=102400,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        attention_bias=True,
        lambda_std_dev=0.1,
        attention_implementation="flash_attention_2",
        
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    current_vocab_size = len(tokenizer)
    
    pad_size = config.vocab_size - current_vocab_size
    if pad_size > 0:
        new_tokens = [f"dummy_{i}" for i in range(pad_size)]
        tokenizer.add_tokens(new_tokens)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if tokenizer.chat_template is None:
        logger.info("No chat template found. Applying a default Jinja template.")
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    
    model = DTForCausalLM(config)

    pretrain_dataset = prepare_pretrain_dataset(
        tokenizer=tokenizer,
        dataset_name_or_path=data_args.dataset_name_or_path,
        dataset_subset=data_args.dataset_subset,
        block_size=data_args.max_length,
        text_column_name=data_args.text_column_name,
    )

    logger.info(f"====== Dataset Info ======")
    logger.info(f"Train dataset:\n{pretrain_dataset}")
    logger.info(f"{next(iter(pretrain_dataset))}")
    logger.info(f"===========================")

    logger.info(f"====== Training Info ======")
    logger.info(f"Training args:\n{training_args}")
    logger.info(f"Model args:\n{model_args}")
    logger.info(f"Data args:\n{data_args}")
    logger.info(f"===========================")

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("====== Model Info ======")
    logger.info(f"Total parameters: {rounded_string(num_params)}")
    logger.info(f"Trainable parameters: {rounded_string(num_trainable_params)} ({100 * num_trainable_params / num_params:.2f}%)")
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Model config:\n{model.config}")
    logger.info(f"========================")

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=pretrain_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        processing_class=tokenizer,
    )
    
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    
    trainer.train(resume_from_checkpoint=checkpoint)

if __name__ == '__main__':
    main()