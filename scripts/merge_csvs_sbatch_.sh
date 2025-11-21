#!/bin/bash
#SBATCH -J merge_csvs_all
#SBATCH -o logs/merge_all_%j.out
#SBATCH -e logs/merge_all_%j.err
#SBATCH -p courses-gpu
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --account=coursesf25

# Activate your youtube_env
source /hpc/home/ah702/work_ah702/venvs/youtube_env/bin/activate

# Go to repo root
cd /hpc/home/ah702/work_ah702/YouTubeOlympics

# Ensure output folders exist
mkdir -p data/processed logs

# Run the merge (this loads everything into memory)
python scripts/merge_csvs.py
