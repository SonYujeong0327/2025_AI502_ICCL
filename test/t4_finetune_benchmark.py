import json
import numpy as np
import random
import torch
import os
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer
from models.dtmodel import DTForCausalLM
from peft import PeftModel
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(SEED)

# model_path = "output_FT/my_sft/checkpoint-1000"
model_path = "output_FT/my_dpo/checkpoint-625"
base_model = DTForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, 
    device_map="cuda", 
    low_cpu_mem_usage=True  
)
#### if your model doesn't have lora weights, just load the base model directly
model = PeftModel.from_pretrained(
    base_model, 
    "output_FT/my_dpo",
)

# model.generation_config.use_cache = False 
# model.config.use_cache = False

tokenizer  = AutoTokenizer.from_pretrained(
    model_path,
)

EVAL_TASKS = [
    "gsm8k",           
    "mmlu",            
    "arc_easy",        
    "hellaswag",       
]

lm = HFLM(
    pretrained=model,
    tokenizer=tokenizer,
    device="cuda",
    dtype="auto",
    trust_remote_code=True,
    batch_size=16,
)

results = evaluator.simple_evaluate(
    model=lm,
    tasks=EVAL_TASKS,
    num_fewshot=0,
    batch_size=16,
    device="cuda",
    log_samples=True,
    limit=100, 
    random_seed=SEED, 
    numpy_random_seed=SEED,  
    torch_random_seed=SEED, 
)

def safe_json(obj):
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json(v) for v in obj]
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    elif hasattr(obj, 'dtype'):
        return str(obj)
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)

with open("finetune_evaluation_results.json", "w") as f:
    json.dump(safe_json(results), f, indent=2, ensure_ascii=False)

print("Evaluation complete. Results saved to finetune_evaluation_results.json")

def extract_metrics(results, task_name, keys):
    task = results["results"].get(task_name, {})
    return {k: task.get(k, None) for k in keys}

report_lines = []
report_lines.append("# Fine-tuned Model Evaluation Report\n")
report_lines.append(f"**Random Seed**: {SEED}")
report_lines.append(f"**Model**: {model_path}")
report_lines.append(f"**Samples per Task**: {100}\n")

# GSM8K
gsm8k_metrics = extract_metrics(results, "gsm8k", ["exact_match", "exact_match_stderr"])
report_lines.append("## GSM8K (Mathematical Reasoning)")
for k, v in gsm8k_metrics.items():
    if v is not None:
        report_lines.append(f"- **{k.upper()}**: {v:.4f}")
report_lines.append("")

# MMLU
mmlu_metrics = extract_metrics(results, "mmlu", ["acc", "acc_stderr"])
report_lines.append("## MMLU (General Knowledge/Reasoning)")
for k, v in mmlu_metrics.items():
    if v is not None:
        report_lines.append(f"- **{k.upper()}**: {v:.4f}")
report_lines.append("")

# ARC Easy
arc_easy_metrics = extract_metrics(results, "arc_easy", ["acc", "acc_norm", "acc_stderr"])
report_lines.append("## ARC Easy (Science/Reasoning)")
for k, v in arc_easy_metrics.items():
    if v is not None:
        report_lines.append(f"- **{k.upper()}**: {v:.4f}")
report_lines.append("")

# HellaSwag
hs_metrics = extract_metrics(results, "hellaswag", ["acc", "acc_norm", "acc_stderr"])
report_lines.append("## HellaSwag (Commonsense Completion)")
for k, v in hs_metrics.items():
    if v is not None:
        report_lines.append(f"- **{k.upper()}**: {v:.4f}")
report_lines.append("")

with open("finetune_evaluation_report.md", "w") as f:
    f.write("\n".join(report_lines))

print("Benchmark report saved to finetune_evaluation_report.md")
