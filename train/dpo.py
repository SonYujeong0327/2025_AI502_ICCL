import os
import warnings
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from models.dtmodel import DTForCausalLM, DTConfig
from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig

# 사용자의 커스텀 모델이 있다면 import (없으면 AutoModel 사용)
# from models.dtmodel import DTForCausalLM 

original_log = DPOTrainer.log

def patched_log(self, logs, start_time=None, **kwargs):
    return original_log(self, logs),

DPOTrainer.log = patched_log

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="./output_FT/DT_sft/checkpoint-1000",
        metadata={"help": "SFT가 완료된 모델의 경로 (또는 베이스 모델)"}
    )
    sft_lora_path: Optional[str] = field(
        default="./output_FT/DT_sft",
        metadata={"help": "SFT LoRA 어댑터 경로 (SFT 때 LoRA를 썼고 Merge 안 했다면 필수)"}
    )

@dataclass
class DataArguments:
    dataset_name_or_path: str = field(
        default="Intel/orca_dpo_pairs", # DPO용 유명 데이터셋
        metadata={"help": "DPO 데이터셋 이름 (prompt, chosen, rejected 컬럼 필요)"}
    )

@dataclass
class LoraArguments:
    use_lora: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules"}
    )

def main():
    warnings.filterwarnings("ignore", message=".*Trainer.tokenizer is now deprecated.*")
    parser = HfArgumentParser((ModelArguments, DataArguments, DPOConfig, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    
    training_args.beta = 0.1

    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Model: {model_args.model_name_or_path}")
    
    # 1. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-4")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 데이터셋 로드 및 전처리
    # DPO는 (prompt, chosen, rejected) 세 가지가 필요합니다.
    dataset = load_dataset(data_args.dataset_name_or_path, split="train[:10000]") # 예시로 1만개만
    
    # 데이터셋 형식에 맞춰 매핑하는 함수 (Intel/orca_dpo_pairs 기준)
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": [f"<|user|>\n{q}\n<|assistant|>\n" for q in samples["question"]],
            "chosen": samples["chosen"],   # 선호하는 답변
            "rejected": samples["rejected"], # 별로인 답변
        }
    
    original_columns = dataset.column_names
    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_columns
    )

    # 3. 모델 로드
    # SFT 된 모델을 베이스로 로드합니다.
    # 만약 SFT를 LoRA로 했고 Merge를 안 했다면, Base + Adapter를 로드해야 합니다.
    model = DTForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attention_implementation="flash_attention_2",
    )
    
    # SFT 어댑터 병합 (선택사항: SFT 결과가 LoRA 파일 형태라면)
    if model_args.sft_lora_path:
        logger.info(f"Loading SFT adapters from {model_args.sft_lora_path}")
        model = PeftModel.from_pretrained(model, model_args.sft_lora_path)
        model = model.merge_and_unload() # 병합해서 하나의 모델로 만듦 (DPO를 위한 베이스)

    # DPO 학습을 위한 새로운 LoRA 설정
    peft_config = None
    if lora_args.use_lora:
        peft_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules.split(","),
            lora_dropout=lora_args.lora_dropout,
            task_type="CAUSAL_LM",
            bias="none",
        )

    # 4. DPO Trainer 설정
    # DPO는 Reference Model이 필요하지만, model만 넣으면 
    # 자동으로 model을 복사해서 ref_model로 사용합니다. (PEFT 사용 시 메모리 절약 가능)
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # None이면 model을 복사해서 사용 (LoRA 사용시 자동으로 처리됨)
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Starting DPO training...")
    trainer.train()
    
    logger.info("Saving model...")
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()