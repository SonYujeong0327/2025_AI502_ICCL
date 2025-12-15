'''
기본 4_finetune_benchmark.py 가 finetune_evaluation_report.md 파일을 잘 생성하지 못해
json만 읽어서 생성하는 코드입니다.
'''

import json

def extract_metrics_robust(results, task_name, keys):
    """
    정확한 키가 없더라도 'acc,none' 같이 변형된 키를 찾아내는 유연한 추출 함수
    """
    task_results = results.get("results", {}).get(task_name, {})
    if not task_results:
        print(f"경고: {task_name} 태스크를 찾을 수 없습니다.")
        return {k: None for k in keys}

    extracted = {}
    for k in keys:
        # 1. 정확히 일치하는 키 확인 (ex: acc)
        if k in task_results:
            extracted[k] = task_results[k]
        else:
            # 2. 변형된 키 확인 (ex: acc,none 또는 acc,strict-match)
            found = False
            for actual_key in task_results.keys():
                if actual_key.startswith(k + ","):
                    extracted[k] = task_results[actual_key]
                    found = True
                    break
            if not found:
                extracted[k] = None
    return extracted

# 1. JSON 파일 로드
file_path = "finetune_evaluation_results.json"
try:
    with open(file_path, "r", encoding="utf-8") as f:
        results = json.load(f)
except FileNotFoundError:
    print(f"에러: {file_path} 파일이 없습니다.")
    exit()

# 2. 메타데이터 (JSON 내부에 정보가 없다면 수동 입력)
seed = results.get("config", {}).get("seed", 42)
model_path = "output_FT/my_dpo/checkpoint-625" # 혹은 results.get("config", {}).get("model_args", "Unknown")
limit = results.get("config", {}).get("limit", 100)

report_lines = []
report_lines.append("# Fine-tuned Model Evaluation Report\n")
report_lines.append(f"**Random Seed**: {seed}")
report_lines.append(f"**Model**: {model_path}")
report_lines.append(f"**Samples per Task**: {limit}\n")

# --- 각 태스크별 데이터 추출 및 기록 ---

# GSM8K
gsm8k_metrics = extract_metrics_robust(results, "gsm8k", ["exact_match", "exact_match_stderr"])
report_lines.append("## GSM8K (Mathematical Reasoning)")
for k, v in gsm8k_metrics.items():
    if v is not None:
        report_lines.append(f"- **{k.upper()}**: {v:.4f}")
report_lines.append("")

# MMLU (MMLU는 하위 태스크가 많아 결과 구조에 따라 확인이 필요합니다)
mmlu_metrics = extract_metrics_robust(results, "mmlu", ["acc", "acc_stderr"])
report_lines.append("## MMLU (General Knowledge/Reasoning)")
for k, v in mmlu_metrics.items():
    if v is not None:
        report_lines.append(f"- **{k.upper()}**: {v:.4f}")
report_lines.append("")

# ARC Easy
arc_easy_metrics = extract_metrics_robust(results, "arc_easy", ["acc", "acc_norm", "acc_stderr"])
report_lines.append("## ARC Easy (Science/Reasoning)")
for k, v in arc_easy_metrics.items():
    if v is not None:
        report_lines.append(f"- **{k.upper()}**: {v:.4f}")
report_lines.append("")

# HellaSwag
hs_metrics = extract_metrics_robust(results, "hellaswag", ["acc", "acc_norm", "acc_stderr"])
report_lines.append("## HellaSwag (Commonsense Completion)")
for k, v in hs_metrics.items():
    if v is not None:
        report_lines.append(f"- **{k.upper()}**: {v:.4f}")
report_lines.append("")

# 3. 파일 저장
with open("finetune_evaluation_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("수정된 Benchmark report가 finetune_evaluation_report.md에 저장되었습니다.")