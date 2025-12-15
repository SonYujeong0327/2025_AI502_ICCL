#!/bin/bash

# SFT가 완료된 모델 경로 (Merge된 모델이거나 Base 모델)
# 만약 SFT 결과가 LoRA 어댑터만 있다면, Base 모델 경로를 적고 --sft_lora_path 옵션을 추가해야 함

export PYTHONPATH=$PYTHONPATH:/home/syj4739/coursework/default_project_release

# MODEL_PATH="./output_FT/my_sft/checkpoint-1000" 
MODEL_PATH="./output_FT/DT_sft/checkpoint-1000" 

CUDA_VISIBLE_DEVICES=1 python -m train.dpo \
    --model_name_or_path $MODEL_PATH \
    --output_dir ./output_FT/DT_dpo \
    --dataset_name_or_path "Intel/orca_dpo_pairs" \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_steps 100 \
    --logging_steps 1 \
    --use_lora True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --report_to wandb \
    --run_name "DT_dpo" \
    --bf16 True