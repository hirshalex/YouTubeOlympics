#!/bin/bash
#SBATCH -J sample_label
#SBATCH -o logs/sample_label_%j.out
#SBATCH -e logs/sample_label_%j.err
#SBATCH -p courses-gpu
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --account=coursesf25

source /hpc/home/ah702/work_ah702/venvs/youtube_env/bin/activate
cd /hpc/home/ah702/work_ah702/YouTubeOlympics
mkdir -p logs data/processed outputs

# adjust quotas as needed:
python -u scripts/sample_for_labeling_stream.py \
  --candidates data/processed/candidates.parquet \
  --all data/processed/ALL_youtube_trending_data.parquet \
  --out_pos data/processed/sampled_pos.parquet \
  --out_neg data/processed/sampled_neg.parquet \
  --pos_per_country 500 \
  --neg_per_country 500 \
  --batch_size 2000
