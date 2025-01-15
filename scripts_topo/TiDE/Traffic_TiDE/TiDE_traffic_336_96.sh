#!/bin/bash

#SBATCH --job-name=TiDE_Traffic_336_96
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=10G
#SBATCH --nodelist=laal1
#SBATCH --partition=laal_3090
#SBATCH --output=/home/jinkwanjang/Time-Series-Library/log/TiDE/S-%x.%j.out

eval "$(conda shell.bash hook)"
conda activate timeSeries

export CUDA_VISIBLE_DEVICES=0

model_name=TiDE

python -u /home/jinkwanjang/Time-Series-Library/run_tide.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /shared/timeSeries/forecasting/base/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_336_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 4 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --des 'Exp' \
  --itr 1 \