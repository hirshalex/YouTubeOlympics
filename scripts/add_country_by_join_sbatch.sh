#!/bin/bash
#SBATCH -J add_country_join
#SBATCH -o logs/add_country_join_%j.out
#SBATCH -e logs/add_country_join_%j.err
#SBATCH -p courses-gpu
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --account=coursesf25

source /hpc/home/ah702/work_ah702/venvs/youtube_env/bin/activate
cd /hpc/home/ah702/work_ah702/YouTubeOlympics
mkdir -p logs data/processed

python -u scripts/add_country_by_join.py \
  --all data/processed/ALL_youtube_trending_data.parquet \
  --candidates data/processed/candidates.parquet \
  --out data/processed/candidates_with_country.parquet \
  --batch_size 5000 \
  --use_signature_if_missing
