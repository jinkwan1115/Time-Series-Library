#!/bin/bash

#SBATCH --job-name=Chronos_Emb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=10G
#SBATCH --nodelist=laal1
#SBATCH --partition=laal_3090
#SBATCH --output=./results/S-%x.%j.out

eval "$(conda shell.bash hook)"
conda activate timeSeries

export CUDA_VISIBLE_DEVICES=0

python -u embed.py \
  --model chronos_bolt \
  --context_len 2048 \
  --data_path /shared/timeSeries/forecasting/base/ETT-small/ETTh1.csv \
  --visualize True