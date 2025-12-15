import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import warnings

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import DataCollatorForLanguageModeling

from models.dtmodel import DTForCausalLM, DTConfig

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    base_model_name_or_path: str = field(
        # default="/mnt/raid5/kangjh/TA2025_2/assignment_2025/output/debug/checkpoint-1380",
        default="./output/dt_dense_wikitext/checkpoint-1000",
        metadata={"help": "Path to the base model for SFT."},
    )

@dataclass
class DataArguments:
    dataset_name_or_path: str = field(
        default="Open-Orca/SlimOrca",
        metadata={"help": "The name or path of the instruction dataset."},
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Max sequence length."},
    )
    packing: bool = field(
        default=True, # Packing 활성화 추천
        metadata={"help": "Use packing to speed up training."},
    )

@dataclass
class LoraArguments:
    use_lora: bool = field(default=True, metadata={"help": "Enable LoRA."})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Target modules. Targeting all linear layers is better."},
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    logging.basicConfig(level=logging.INFO)
    
    # 1. Tokenizer 로드 및 설정
    # tokenizer = AutoTokenizer.from_pretrained(model_args.base_model_name_or_path)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right" # Packing 사용 시 right padding이 일반적
    
    tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-4")
    
    current_vocab_size = len(tokenizer)

    # Dummy tokens for empty vocab slots (if vocab_size > current tokenizer size)
    pad_size = 102400 - current_vocab_size
    if pad_size > 0:
        new_tokens = [f"dummy_{i}" for i in range(pad_size)]
        tokenizer.add_tokens(new_tokens)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if tokenizer.chat_template is None:
        logger.info("No chat template found. Applying a default Jinja template.")
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"

    # 2. 모델 로드
    model = DTForCausalLM.from_pretrained(
        model_args.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        attention_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )

    # 3. LoRA 설정 (모든 리니어 레이어 타겟팅)
    peft_config = None
    if lora_args.use_lora:
        if lora_args.lora_target_modules and len(lora_args.lora_target_modules) == 1:
            if "," in lora_args.lora_target_modules[0]:
                lora_args.lora_target_modules = lora_args.lora_target_modules[0].split(",")

        peft_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=lora_args.lora_target_modules, # 수정된 리스트가 들어감
            task_type="CAUSAL_LM",
            bias="none",
        )

    # 4. 데이터셋 포맷팅 함수 (Tokenizing은 Trainer가 수행)
    def formatting_prompts_func(example):
        output_texts = []
        
        # 배치의 첫 번째 샘플을 확인하여 컬럼 키를 파악합니다.
        # 이렇게 하면 데이터셋이 바뀌어도 코드가 터지지 않습니다.
        keys = example.keys()
        batch_size = len(example[list(keys)[0]])

        for i in range(batch_size):
            # Case A: 'conversations' 컬럼이 있는 경우 (OpenHermes, SlimOrca 등 최신 데이터셋)
            if 'conversations' in example:
                conversations = example['conversations'][i]
                # conversations는 보통 [{'from': 'human', 'value': '...'}, {'from': 'gpt', 'value': '...'}] 형태입니다.
                text = ""
                for message in conversations:
                    role = "user" if message['from'] in ['human', 'user'] else "assistant"
                    content = message['value']
                    text += f"<|{role}|>\n{content}\n"
                text += tokenizer.eos_token # 대화 끝에 EOS 토큰 추가

            # Case B: 'messages' 컬럼이 있는 경우 (HuggingFace 표준 Chat 포맷)
            elif 'messages' in example:
                messages = example['messages'][i]
                text = tokenizer.apply_chat_template(messages, tokenize=False)

            # Case C: 기존 'instruction', 'output' 구조 (Alpaca, Dolly 등)
            elif 'instruction' in example:
                instruction = example['instruction'][i]
                output = example['output'][i] if 'output' in example else example['response'][i]
                input_text = example['input'][i] if 'input' in example and example['input'][i] else ""
                
                if input_text:
                    text = f"<|user|>\n{instruction}\nInput:\n{input_text}\n<|assistant|>\n{output}"
                else:
                    text = f"<|user|>\n{instruction}\n<|assistant|>\n{output}"

            # Case D: 만약 컬럼을 못 찾겠다면 (디버깅용)
            else:
                # 사용 가능한 컬럼명을 출력하고 에러를 냅니다.
                available_columns = list(example.keys())
                raise ValueError(f"지원되지 않는 데이터셋 형식입니다. 사용 가능한 컬럼: {available_columns}")

            output_texts.append(text)

        return output_texts

    dataset = load_dataset(data_args.dataset_name_or_path, split="train")

    # 5. Data Collator (Response 부분만 학습하도록 설정)
    # response_template은 사용자가 답변을 시작하기 직전의 문자열이어야 함
    response_template = "<|assistant|>\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, 
        tokenizer=tokenizer
    )

    # 6. SFTTrainer 사용
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=data_args.max_length,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator, # Packing을 쓴다면 collator는 None으로 두는 게 일반적일 수 있으나, TRL 최신 버전은 지원함.
        # Packing 사용 시 주의: Packing=True면 collator 무시될 수 있음 (문서 확인 필요). 
        # Instruction Tuning 효과를 극대화하려면 packing=False + collator 사용 추천.
        # 속도가 중요하다면 packing=True (이 경우 collator 제거).
        # packing=data_args.packing, 
        packing=False, 
    )

    trainer.train()
    
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()