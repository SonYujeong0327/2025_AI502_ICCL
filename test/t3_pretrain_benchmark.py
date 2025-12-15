import torch
import time
import json
import os
import tracemalloc
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from models.dtmodel import DTForCausalLM

CONFIG = {
    "model_name": "dt",  
    "model_path": "output_FT/my_dpo/checkpoint-625",
    "seed": 42,  
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_file": "evaluation_results.json",
    "performance": {
        "num_runs": 10, 
        "max_new_tokens": 50
    },
    "mmlu": {
        "dataset_name": "cais/mmlu",
        "dataset_config": "all",
        "num_samples": 100 
    }
}

def set_seed(seed):
    """
    Sets all random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    return total, trainable

def human_readable(n):
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.3f}B"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.3f}M"
    else:
        return str(n)

def evaluate_perplexity(model, tokenizer, device):
    """
    Calculates Perplexity on a sample from WikiText-2-raw-v1 using a standard
    sliding-window with stride and proper next-token shifting.
    """
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
    split = "test"
    num_samples = 100

    print(f"Starting evaluation: Perplexity (PPL) with {dataset_name} ({num_samples} samples)...")
    model.eval()

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    dataset = dataset.shuffle(seed=CONFIG["seed"]).select(range(num_samples))
    text = "\n\n".join([sample["text"] for sample in dataset if sample["text"].strip() != ""])
    if not text:
        print("Warning: No valid text found in sampled data for perplexity evaluation. Returning NaN.")
        return float("nan")
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(device)
    seq_len = input_ids.size(1)

    max_length = getattr(model.config, "max_position_embeddings", 1024)
    stride = max_length // 2

    nll_sum = 0.0
    n_tokens = 0

    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - begin_loc
            if trg_len <= 1:
                break

            if begin_loc == 0:
                valid_len = trg_len - 1
            else:
                valid_len = end_loc - begin_loc - (max_length - stride)
                valid_len = max(valid_len, 0)

            input_ids_segment = input_ids[:, begin_loc:end_loc]
            labels = input_ids_segment.clone()
            labels[:, 0] = -100

            outputs = model(input_ids_segment, labels=labels)
            if valid_len > 0:
                nll_sum += outputs.loss.item() * valid_len
                n_tokens += valid_len

            if end_loc == seq_len:
                break

    if n_tokens == 0:
        print("Warning: No valid tokens counted. Returning NaN.")
        return float("nan")

    ppl = float(torch.exp(torch.tensor(nll_sum / n_tokens)))
    print(f"Perplexity evaluation finished: {ppl:.4f}")
    return ppl

def evaluate_performance(model, tokenizer, config, device):
    """
    Measures average Latency and Memory usage over multiple inference runs.
    Uses parts of the Wikitext dataset as prompts.
    """
    print(f"Starting evaluation: Performance (Latency & Memory) over {config['num_runs']} runs...")
    model.eval()

    # Load prompts for Latency/Memory measurement
    prompts_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    prompts_dataset = prompts_dataset.shuffle(seed=CONFIG["seed"]).select(range(config['num_runs']))
    prompts = [sample['text'] for sample in prompts_dataset if sample['text'].strip() != '']

    latencies = []
    peak_memories = []

    for prompt in tqdm(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        tracemalloc.start()
        start_time = time.time()

        with torch.no_grad():
            model.generate(
                input_ids,
                max_new_tokens=config['max_new_tokens'],
                do_sample=False
            )
        
        end_time = time.time()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        duration = end_time - start_time
        tokens_per_second = config['max_new_tokens'] / duration
        peak_memory_mb = peak / 1024 / 1024

        latencies.append(tokens_per_second)
        peak_memories.append(peak_memory_mb)

    avg_tokens_per_sec = np.mean(latencies)
    std_tokens_per_sec = np.std(latencies)
    avg_peak_memory = np.mean(peak_memories)
    std_peak_memory = np.std(peak_memories)

    print("Performance evaluation finished:")
    print(f"  - Avg Tokens/sec: {avg_tokens_per_sec:.4f} (±{std_tokens_per_sec:.4f})")
    print(f"  - Avg Peak Memory (MB): {avg_peak_memory:.4f} (±{std_peak_memory:.4f})")

    return {
        "avg_tokens_per_second": avg_tokens_per_sec,
        "std_tokens_per_second": std_tokens_per_sec,
        "avg_peak_memory_mb": avg_peak_memory,
        "std_peak_memory_mb": std_peak_memory,
        "num_runs": config['num_runs']
    }

def evaluate_mmlu(model, tokenizer, num_samples, device):
    """
    Evaluates Zero-shot performance on the MMLU dataset.
    It compares the probabilities the model assigns to the answer choices ('A', 'B', 'C', 'D')
    and takes the highest one as the prediction.
    """
    print(f"Starting evaluation: MMLU ({num_samples} samples)...")
    model.eval()

    # Load MMLU dataset (using the test split)
    # To reduce evaluation time, shuffle the dataset and use a subset.
    # Fix the seed to ensure the same samples are always drawn.
    dataset = load_dataset(CONFIG["mmlu"]["dataset_name"], CONFIG["mmlu"]["dataset_config"], split="test")
    dataset = dataset.shuffle(seed=CONFIG["seed"]).select(range(num_samples))

    choices = ["A", "B", "C", "D"]
    # The token ID for 'A' and ' A' can differ, so get the ID with a preceding space.
    choice_token_ids = {c: tokenizer.encode(f' {c}', add_special_tokens=False)[0] for c in choices}

    correct_predictions = 0

    for sample in tqdm(dataset):
        question = sample['question']
        options = sample['choices']
        answer_idx = sample['answer'] # 0, 1, 2, 3

        # Construct Zero-shot prompt
        prompt = f"Question: {question}\n"
        for i, option in enumerate(options):
            prompt += f"{choices[i]}. {option}\n"
        prompt += "Answer:"
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Extract logits for the choice tokens from the last token's logits.
        last_token_logits = logits[0, -1, :]
        choice_logits = {
            c: last_token_logits[choice_token_ids[c]].item() for c in choices
        }

        # Determine the model's prediction as the choice with the highest logit.
        prediction = max(choice_logits, key=choice_logits.get)
        predicted_idx = choices.index(prediction)

        if predicted_idx == answer_idx:
            correct_predictions += 1
            
    accuracy = correct_predictions / num_samples
    print(f"MMLU evaluation finished: Accuracy = {accuracy:.4f}")
    return {
        "accuracy": accuracy,
        "num_samples": num_samples
    }

def main():
    """
    Main evaluation pipeline
    """
    set_seed(CONFIG['seed'])
    print(f"Fixed random seed: {CONFIG['seed']}")

    print(f"Loading model: {CONFIG['model_path']}...")

    model = DTForCausalLM.from_pretrained(
        CONFIG['model_path'],
        dtype=torch.bfloat16,
        device_map="auto",
        attention_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_path'])

    '''model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')'''

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model.to(CONFIG['device'])
    print(f"Model loaded to {CONFIG['device']}.")
    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters: total={total_params} ({human_readable(total_params)}), "
          f"trainable={trainable_params} ({human_readable(trainable_params)})")

    results = {}
    results['model_name'] = CONFIG['model_path']
    results['evaluation_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
    results['model_params'] = {
        "total": total_params,
        "total_human": human_readable(total_params),
        "trainable": trainable_params,
        "trainable_human": human_readable(trainable_params)
    }


    results['perplexity'] = evaluate_perplexity(model, tokenizer, CONFIG['device'])
    results['performance'] = evaluate_performance(
        model, tokenizer, CONFIG['performance'], CONFIG['device']
    )

    results['mmlu'] = evaluate_mmlu(model, tokenizer, CONFIG['mmlu']['num_samples'], CONFIG['device'])

    with open(CONFIG['output_file'], 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nAll evaluations are complete. Results have been saved to '{CONFIG['output_file']}'.")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()