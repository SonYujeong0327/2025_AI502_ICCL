import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from models.dtmodel import DTForCausalLM, DTConfig
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    base_model_name_or_path: str = field(
        # default="/mnt/raid5/kangjh/TA2025_2/assignment_2025/output/debug/checkpoint-1380", 
        default="./output_onlyDT/checkpoint-1000",
        metadata={"help": "Path to the base model for SFT."},
    )


@dataclass
class DataArguments:
    dataset_name_or_path: str = field(
        # default="databricks/databricks-dolly-15k",
        default="Open-Orca/SlimOrca",
        metadata={"help": "The name or path of the instruction dataset."},
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Max sequence length. Sequences will be right-padded (and possibly truncated)."},
    )


@dataclass
class LoraArguments:
    use_lora: bool = field(default=True, metadata={"help": "Enable LoRA."})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_target_modules: str = field(
        default="q_proj,v_proj",
        metadata={"help": "Comma separated list of module names to apply LoRA to."},
    )


def rounded_string(target: int) -> str:
    if target < 2**10:
        return str(target)
    elif target < 2**20:
        return f"{target / 2**10:.2f}K"
    elif target < 2**30:
        return f"{target / 2**20:.2f}M"
    elif target < 2**40:
        return f"{target / 2**30:.2f}G"
    else:
        return f"{target / 2**40:.2f}T"


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-4")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "right" 
    
    if tokenizer.chat_template is None:
        logger.info("No chat template found. Applying a default Zephyr-style template.")
        tokenizer.chat_template = (
            "{% for message in messages %}\n"
            "{% if message['role'] == 'user' %}\n"
            "{{ '<|user|>\n' + message['content'] + eos_token }}\n"
            "{% elif message['role'] == 'assistant' %}\n"
            "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
            "{% endif %}\n"
            "{% endfor %}"
        )


    model = DTForCausalLM.from_pretrained(
        model_args.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        attention_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )

    if lora_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=lora_args.lora_target_modules.split(","),
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        logger.info("Applied LoRA to the model.")

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("====== Model Info ======")
    logger.info(f"Total parameters: {rounded_string(num_params)}")
    logger.info(f"Trainable parameters: {rounded_string(num_trainable_params)} ({100 * num_trainable_params / num_params:.2f}%)")
    logger.info(f"Model architecture:\n{model}")
    logger.info("========================")

    def create_prompt_with_template(example):
        user_content = example['instruction']
        if 'context' in example and example['context']:
            user_content += f"\n\nContext:\n{example['context']}"
        
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example['response']}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    dataset = load_dataset(data_args.dataset_name_or_path, split="train")

    
    tokenized_dataset = dataset.map(
        create_prompt_with_template,
        remove_columns=dataset.column_names,
    ).map(
        lambda x: tokenizer(x["text"], max_length=data_args.max_length, truncation=True),
        batched=True
    )
    
    logger.info(f"====== Dataset Info ======")
    logger.info(f"Train dataset:\n{tokenized_dataset}")
    logger.info(f"Sample data:\n{tokenized_dataset[0]}")
    logger.info(f"Decoded sample:\n{tokenizer.decode(tokenized_dataset[0]['input_ids'])}")
    logger.info("===========================")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    final_checkpoint_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.save_model(final_checkpoint_dir)
    logger.info(f"Final LoRA adapter saved to {final_checkpoint_dir}")


if __name__ == "__main__":
    main()