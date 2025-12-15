#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --nodelist=server2
#SBATCH --output=slurm_logs/pretrain%j.out
#SBATCH --error=slurm_logs/pretrain%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=100:00:00

# cd /mnt/raid5/kangjh/TA2025_2/release/assignment_2025
# source .venv/bin/activate

CUDA_VISIBLE_DEVICES=2

export PYTHONPATH=$PYTHONPATH:/home/syj4739/coursework/default_project_release
export WANDB_RUN_ID="c6vuyz6a"
export WANDB_RESUME="allow"

echo "Python path: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo $CUDA_VISIBLE_DEVICES

#CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch \
# accelerate launch \
#     --config_file config/debug.yaml \
#     --main_process_port 43489 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m train_vanilla.pretrain \
    --output_dir ./output_vanilla \
    --max_steps 1000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --logging_steps 1 \
    --include_tokens_per_second \
    --include_num_input_tokens_seen \
    --save_steps 20 \
    --save_total_limit 3 \
    --dataset_name_or_path wikitext \
    --dataset_subset "wikitext-2-raw-v1" \
    --report_to wandb \
    --run_name "vanilla-baseline-1k" \
    --resume_from_checkpoint "./output_vanilla/checkpoint-900"
    2>&1 | tee train_vanilla_log.txt
