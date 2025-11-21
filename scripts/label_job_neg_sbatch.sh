#!/bin/bash
#SBATCH -J llm_label_neg
#SBATCH -o logs/label_neg_%j.out
#SBATCH -e logs/label_neg_%j.err
#SBATCH -p courses-gpu
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=coursesf25

source /hpc/home/ah702/work_ah702/venvs/youtube_env/bin/activate
cd /hpc/home/ah702/work_ah702/YouTubeOlympics

export PYTHONUNBUFFERED=1
python -u scripts/label_with_llm.py \
  --input data/processed/sampled_neg.parquet \
  --output data/processed/labels_neg.csv \
  --model Qwen/Qwen2-1.5B-Instruct \
  --maxrows 3000 \
  --prompt_file scripts/labeling_prompt_examples.md \
  --batch_size 1 \
  --save_every 100