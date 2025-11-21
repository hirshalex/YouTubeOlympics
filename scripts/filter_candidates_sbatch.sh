#!/bin/bash
#SBATCH -J filter_cand_mem
#SBATCH -o logs/filter_cand_%j.out
#SBATCH -e logs/filter_cand_%j.err
#SBATCH -p courses-gpu
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --account=coursesf25

source /hpc/home/ah702/work_ah702/venvs/youtube_env/bin/activate
cd /hpc/home/ah702/work_ah702/YouTubeOlympics


export PYTHONUNBUFFERED=1
python -u scripts/filter_candidates_multilang.py --output data/processed/candidates.parquet
