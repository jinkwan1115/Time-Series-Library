#!/bin/bash

#SBATCH --job-name=FreTS_ETTm1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=80000MB
#SBATCH --cpus-per-task=16
#SBATCH --output=./folder/S-%x.%j.out

eval "$(conda shell.bash hook)"
conda activate timeSeries


export CUDA_VISIBLE_DEVICES=0

model_name=FreTS

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1