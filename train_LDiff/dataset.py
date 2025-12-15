import logging
from itertools import chain
from datasets import load_dataset

logger = logging.getLogger(__name__)

def prepare_pretrain_dataset(
    tokenizer,
    dataset_name_or_path,
    dataset_subset=None,
    block_size=1024,
    text_column_name="text",
    num_proc=4,  # 데이터 처리에 사용할 CPU 코어 수
):
    """
    Prepares a dataset for Causal Language Modeling (Pretraining).
    1. Loads the dataset.
    2. Tokenizes the text.
    3. Groups texts into chunks of `block_size`.
    """
    
    # 1. Load Dataset
    logger.info(f"Loading dataset: {dataset_name_or_path} (subset: {dataset_subset})")
    if dataset_subset:
        raw_datasets = load_dataset(dataset_name_or_path, dataset_subset)
    else:
        raw_datasets = load_dataset(dataset_name_or_path)

    # 데이터셋 구조 확인 (train split이 없으면 생성)
    if "train" not in raw_datasets:
        # 데이터셋이 분할되어 있지 않은 경우 처리
        raise ValueError(f"Dataset {dataset_name_or_path} does not have a 'train' split.")

    column_names = raw_datasets["train"].column_names
    
    # 2. Tokenize
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    logger.info("Tokenizing dataset...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=column_names, # 원본 텍스트 제거하여 메모리 절약
        desc="Running tokenizer on dataset",
    )

    # 3. Grouping (Concatenation & Chunking)
    # 텍스트들을 전부 이어 붙인 뒤, block_size만큼 잘라서 예제를 만듭니다.
    def group_texts(examples):
        # 모든 텍스트를 하나로 이어 붙임
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # 마지막 자투리가 block_size보다 작으면 버림
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            
        # block_size 단위로 자르기
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        
        # Causal LM 학습을 위해 labels는 input_ids와 동일하게 설정
        # (Hugging Face Trainer가 내부적으로 shift 시킴)
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info(f"Grouping texts into chunks of {block_size}...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    # train dataset만 반환 (pretrain.py 구조상)
    return lm_datasets["train"]