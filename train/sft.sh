#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --nodelist=server2
#SBATCH --output=slurm_logs/sft%j.out
#SBATCH --error=slurm_logs/sft%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=100:00:00

# cd /mnt/raid5/kangjh/TA2025_2/assignment_2025
# source .venv/bin/activate

export PYTHONPATH=$PYTHONPATH:/home/syj4739/coursework/default_project_release

CUDA_VISIBLE_DEVICES=3

echo "Python path: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo $CUDA_VISIBLE_DEVICES

# my version
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m train.sft \
#     --output_dir ./output_DT/my_sft \
#     --max_steps 1000 \
#     --learning_rate 2e-5 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --logging_steps 1 \
#     --save_steps 20 \
#     --save_total_limit 3 \
#     --use_lora True \
#     --lora_r 64 \
#     --lora_alpha 128 \
#     --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
#     --lora_dropout 0.05 \
#     --report_to wandb \
#     --run_name "sft-onlyDT" \
#     2>&1 | tee sft_onlyDT_log.txt

# original version
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m train.sft \
#     --output_dir ./output_FT/orig_sft \
#     --max_steps 1000 \
#     --learning_rate 2e-5 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --logging_steps 10 \
#     --save_steps 100 \
#     --save_total_limit 3 \
#     --use_lora True \
#     --lora_r 16 \
#     --lora_alpha 32 \
#     --lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
#     --report_to wandb \
#     --run_name "orig_sft"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m train.sft \
    --output_dir ./output_FT/DT_sft \
    --max_steps 1000 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 3 \
    --use_lora True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules "q_proj,v_proj,k_proj,o_proj" \
    --report_to wandb \
    --run_name "DT_sft"
